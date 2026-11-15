// OlingoBridge: resident Swift process that owns the microphone and the speakers.
// Argmax WhisperKit (speech to text) and TTSKit (text to speech) are loaded once, from local folders only
// (no download, no network), and stay in memory for the whole call.
//
// stdin : one JSON command per line.   stdout: one JSON event per line (nothing else).   stderr: logs.
//   {"cmd":"say","id":"7","text":"...","lang":"en","rate":1.0,"save":"/abs/x.wav","mute":false}
//   {"cmd":"play","id":"8","file":"/abs/x.wav","rate":1.0}
//        Speech is pipelined: an item is synthesised while the previous one is still playing and is queued right
//        behind it. -> {"event":"start","id"} when its first sound is audible, {"event":"said","id"} when it ends.
//   {"cmd":"listen","lang":"en","alt":"zh","wait":10,"max":12,"filler":"/abs/hmm.wav","ptt":false}
//        Waits until playback is over, then listens. lang "auto" = detect among "langs". "alt" = second language to
//        try when the first transcription is unsure. "filler" is played the instant the learner stops talking.
//        -> listening, speech_start, speech_end, heard {text, lang, words:[{w,p}], audio_s, ms}
//   {"cmd":"endpoint"}   push-to-talk released: end the utterance now
//   {"cmd":"stop"}       cancel everything that is playing, queued or listening, at once
//   {"cmd":"transcribe","file":"/abs/x.wav","lang":"en"}  -> transcribed {text}      (self-tests)
//   listen also takes "voice":"enroll" (learn the speaker from this utterance) or "voice":"check" with
//        "voice_min": utterances whose voice is less similar than that to the learned one are dropped unheard.
//        Argmax SpeakerKit makes the voice embedding. It is held in memory only; {"cmd":"forget_voice"} erases it.
//   {"cmd":"audio_mode","agent":true}   sound now goes to, and comes from, the Olingo device instead of this Mac:
//        speech is not played here but emitted as {"event":"spk","pcm":"<base64 PCM16 24 kHz>"} (and "spk_flush"
//        when it must stop); the microphone is fed with {"cmd":"mic","pcm":"<base64 PCM16 16 kHz>"}.
//   {"cmd":"ping"} -> pong      {"cmd":"quit"}
// Microphone audio only ever lives in memory.
import AVFoundation
import CoreML
import Foundation
import SpeakerKit
import TTSKit
import WhisperKit

// MARK: - IO
let outLock = NSLock()
func emit(_ d: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: d) else { return }
    outLock.lock(); FileHandle.standardOutput.write(data + Data([10])); outLock.unlock()
}
func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func ms(_ t0: Date) -> Int { Int(Date().timeIntervalSince(t0) * 1000) }
let env = ProcessInfo.processInfo.environment
func envInt(_ k: String, _ d: Int) -> Int { Int(env[k] ?? "") ?? d }

/// `stop` bumps the epoch: every command that was read before it is cancelled.
final class Ctl: @unchecked Sendable {
    private let lock = NSLock(); private var _epoch = 0, _endpoint = false
    var epoch: Int { lock.lock(); defer { lock.unlock() }; return _epoch }
    func bump() { lock.lock(); _epoch += 1; lock.unlock() }
    var endpoint: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _endpoint }
        set { lock.lock(); _endpoint = newValue; lock.unlock() }
    }
}
let ctl = Ctl()

/// Where sound is played and heard: this Mac's own speakers and microphone, or the Olingo device (the agent).
final class AudioMode: @unchecked Sendable {
    private let lock = NSLock(); private var _agent = false, _mic = false
    /// Speech goes to the device instead of this Mac's speakers.
    var agent: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _agent }
        set { lock.lock(); _agent = newValue; lock.unlock() }
    }
    /// The microphone is the device's. False with agent true = the device speaks, this Mac listens (used when the
    /// link is too slow to carry the device's microphone).
    var mic: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _mic }
        set { lock.lock(); _mic = newValue; lock.unlock() }
    }
}
let audioMode = AudioMode()
let agentDelay = Double(env["AGENT_AUDIO_DELAY_MS"] ?? "").map { $0 / 1000 } ?? 0.3     // link + playback on the device

func rms(_ x: UnsafePointer<Float>, _ n: Int) -> Float {
    var s: Float = 0
    for i in 0..<n { s += x[i] * x[i] }
    return n > 0 ? (s / Float(n)).squareRoot() : 0
}

// MARK: - Player: one continuous queue for streamed speech and cached wav files
final class Player: @unchecked Sendable {
    let engine = AVAudioEngine()
    let node = AVAudioPlayerNode()
    let pitch = AVAudioUnitTimePitch()
    let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!
    private let lock = NSLock()
    private var pending = 0                      // buffers scheduled and not yet played
    private var gen = 0                          // bumped by stopNow(): callbacks of cancelled buffers are ignored
    private var queueEnd = Date.distantPast      // wall clock at which everything scheduled so far has been played
    private var inflight: [String] = []          // ids whose `said` has not been sent yet
    private var lastLevel = Date.distantPast
    private var warned = false

    init() {
        engine.attach(node); engine.attach(pitch)
        pitch.bypass = true
        wire()
        pitch.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, _ in
            guard let self, self.node.isPlaying, let d = buf.floatChannelData?[0] else { return }
            if Date().timeIntervalSince(self.lastLevel) >= 0.07 {
                self.lastLevel = Date()
                emit(["event": "level", "v": min(1.0, Double(rms(d, Int(buf.frameLength))) * 6)])
            }
        }
        // Output device, sample rate or route changed (headphones, mute key, another app): the engine stops itself.
        NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
            log("player: audio configuration changed, restarting")
            self?.wire(); _ = self?.ensure()
        }
        _ = ensure()
    }

    private func wire() {
        engine.connect(node, to: pitch, format: fmt)
        engine.connect(pitch, to: engine.mainMixerNode, format: fmt)
    }

    @discardableResult func ensure() -> Bool {
        if engine.isRunning { return true }
        engine.prepare()
        do { try engine.start(); warned = false; return true } catch {
            log("player engine start failed: \(error)")
            if !warned {
                warned = true
                emit(["event": "error", "code": "audio_output",
                      "message": "macOS refused to start the speakers (this happens when memory is nearly full)"])
            }
            return false
        }
    }

    /// Seconds of audio still queued ahead of anything scheduled now.
    var lead: Double { lock.lock(); defer { lock.unlock() }; return max(0, queueEnd.timeIntervalSinceNow) }
    var busy: Bool {
        lock.lock(); defer { lock.unlock() }
        return pending > 0 && Date() < queueEnd.addingTimeInterval(1.0 + agentDelay)   // never wait on callbacks that will not come
    }
    func waitIdle(_ epoch: Int) async {
        while busy && ctl.epoch == epoch { try? await Task.sleep(nanoseconds: 10_000_000) }
    }
    /// The rate applies to everything in the queue, so it only changes once the queue is empty.
    func setRate(_ r: Float, _ epoch: Int) async {
        if abs(pitch.rate - r) < 0.005 { return }
        await waitIdle(epoch)
        pitch.rate = r
        pitch.bypass = abs(r - 1.0) < 0.005              // at normal speed the audio goes straight through
    }

    /// The device plays the sound. Here it is only sent, and the completion callbacks fire on a clock that follows
    /// what the device is playing (plus the delay of the link), so "it has finished speaking" stays true.
    private func scheduleRemote(_ samples: [Float], done: (@Sendable () -> Void)?) {
        guard !samples.isEmpty else { done?(); return }
        lock.lock()
        let g = gen
        pending += 1
        let now = Date()
        let start = queueEnd > now ? queueEnd : now
        let end = start.addingTimeInterval(Double(samples.count) / 24000.0)
        queueEnd = end
        lock.unlock()
        var pcm = Data(capacity: samples.count * 2)
        var peak: Float = 0
        for s in samples {
            let c = max(-1, min(1, s))
            peak = max(peak, abs(c))
            var v = Int16(c * 32767).littleEndian
            withUnsafeBytes(of: &v) { pcm.append(contentsOf: $0) }
        }
        emit(["event": "spk", "pcm": pcm.base64EncodedString()])
        let level = min(1.0, Double(peak) * 2)
        DispatchQueue.global().asyncAfter(deadline: .now() + max(0, start.timeIntervalSinceNow) + agentDelay) {
            emit(["event": "level", "v": level])
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + max(0, end.timeIntervalSinceNow) + agentDelay) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let live = g == self.gen
            if live { self.pending = max(0, self.pending - 1) }
            self.lock.unlock()
            if live { done?() }
        }
    }

    func schedule(_ samples: [Float], done: (@Sendable () -> Void)? = nil) {
        if audioMode.agent { scheduleRemote(samples, done: done); return }
        guard !samples.isEmpty, ensure(),
              let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(samples.count)) else { done?(); return }
        buf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buf.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        lock.lock()
        let g = gen
        pending += 1
        let now = Date()
        queueEnd = (queueEnd > now ? queueEnd : now).addingTimeInterval(Double(samples.count) / 24000.0 / Double(max(pitch.rate, 0.25)))
        lock.unlock()
        node.scheduleBuffer(buf, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            self.lock.lock()
            let live = g == self.gen
            if live { self.pending = max(0, self.pending - 1) }
            self.lock.unlock()
            if live { done?() }
        }
        if !node.isPlaying { node.play() }
    }

    func begin(_ id: String) { lock.lock(); inflight.append(id); lock.unlock() }
    /// Sends `said` exactly once per id.
    func finish(_ id: String, _ extra: [String: Any]) {
        lock.lock()
        guard let i = inflight.firstIndex(of: id) else { lock.unlock(); return }
        inflight.remove(at: i)
        lock.unlock()
        var e = extra; e["event"] = "said"; e["id"] = id
        emit(e)
    }
    /// If the audio hardware stalls, the completion callback never comes: report the item as finished anyway.
    func watchdog(_ id: String) {
        lock.lock(); let end = queueEnd; lock.unlock()
        Task.detached { [weak self] in
            let wait = max(0, end.timeIntervalSinceNow) + 2.0
            try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
            guard let self else { return }
            self.lock.lock(); let still = self.inflight.contains(id); self.lock.unlock()
            if still {
                emit(["event": "error", "code": "audio_output", "message": "the speakers stalled"])
                self.finish(id, ["stalled": true])
            }
        }
    }
    func stopNow() {
        lock.lock()
        gen += 1; pending = 0; queueEnd = Date()
        let ids = inflight; inflight = []
        lock.unlock()
        node.stop()
        if audioMode.agent { emit(["event": "spk_flush"]) }      // the device must fall silent too
        for id in ids { emit(["event": "said", "id": id, "stopped": true]) }
    }

    func load(_ path: String) -> [Float]? {
        guard let f = try? AVAudioFile(forReading: URL(fileURLWithPath: path)),
              let buf = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length)),
              (try? f.read(into: buf)) != nil else { return nil }
        var out = buf
        if buf.format != fmt {
            guard let conv = AVAudioConverter(from: buf.format, to: fmt),
                  let o = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(Double(buf.frameLength) * 24000 / buf.format.sampleRate) + 64) else { return nil }
            var consumed = false
            var err: NSError?
            conv.convert(to: o, error: &err) { _, st in
                if consumed { st.pointee = .noDataNow; return nil }
                consumed = true; st.pointee = .haveData; return buf
            }
            if err != nil { return nil }
            out = o
        }
        guard let d = out.floatChannelData?[0] else { return nil }
        return Array(UnsafeBufferPointer(start: d, count: Int(out.frameLength)))
    }
}

// MARK: - Mic: always on; collects only while a `listen` is active; end-pointing by energy
final class Mic: @unchecked Sendable {
    let engine = AVAudioEngine()
    let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    private var converter: AVAudioConverter?
    private let lock = NSLock()
    private let frame = 320                      // 20 ms @ 16 kHz
    private var acc: [Float] = []
    private var samples: [Float] = []
    private var collecting = false, vad = true
    private var speechStarted = false, done = false
    private var speechFrames = 0, silenceFrames = 0, speechStartIdx = 0
    private var sinceListen = 0, maxSamples = 0, noSpeechMs = 8000
    private var floor: Float = 0.004
    private var observing = false, warned = false
    private var lastLevel = Date.distantPast
    private var frames = 0
    private var minLevel: Float = 0              // set per listen: quieter sound than this is not speech
    private var loudSum: Float = 0, loudN = 0    // mean loudness of the frames that counted as speech
    var level: Double { lock.lock(); defer { lock.unlock() }; return loudN > 0 ? Double(loudSum / Float(loudN)) : 0 }
    let silenceEndMs = envInt("VAD_SILENCE_MS", 700)   // once the learner has stopped, do not keep them waiting
    let minThr = Float(env["VAD_MIN_THR"] ?? "") ?? 0.012
    var framesSeen: Int { lock.lock(); defer { lock.unlock() }; return frames }

    func start() throws {
        engine.inputNode.removeTap(onBus: 0)
        let inFmt = engine.inputNode.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: inFmt, to: target)
        engine.inputNode.installTap(onBus: 0, bufferSize: 2048, format: inFmt) { [weak self] buf, _ in self?.process(buf) }
        engine.prepare()
        try engine.start()
        log("mic: \(inFmt.sampleRate) Hz, \(inFmt.channelCount) ch")
        if !observing {
            observing = true
            NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
                log("mic: audio configuration changed, restarting")
                _ = self?.ensureRunning()
            }
        }
    }
    @discardableResult func ensureRunning() -> Bool {
        if audioMode.mic || engine.isRunning { return true }
        do { try start(); warned = false; return true } catch {
            log("mic start failed: \(error)")
            if !warned {
                warned = true
                emit(["event": "error", "code": "audio_input",
                      "message": "macOS refused to start the microphone (this happens when memory is nearly full)"])
            }
            return false
        }
    }

    /// Microphone audio from the Olingo device (16 kHz mono). It takes the place of this Mac's microphone.
    func feed(_ s: [Float]) {
        lock.lock(); defer { lock.unlock() }
        acc.append(contentsOf: s)
        while acc.count >= frame {
            let f = Array(acc.prefix(frame)); acc.removeFirst(frame)
            handle(f)
        }
    }

    private func process(_ buf: AVAudioPCMBuffer) {
        if audioMode.mic { return }                      // the device is the microphone now
        guard let conv = converter else { return }
        let cap = AVAudioFrameCount(Double(buf.frameLength) * 16000 / buf.format.sampleRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap) else { return }
        var consumed = false
        var err: NSError?
        conv.convert(to: out, error: &err) { _, st in
            if consumed { st.pointee = .noDataNow; return nil }
            consumed = true; st.pointee = .haveData; return buf
        }
        guard err == nil, let d = out.floatChannelData?[0] else { return }
        lock.lock(); defer { lock.unlock() }
        acc.append(contentsOf: UnsafeBufferPointer(start: d, count: Int(out.frameLength)))
        while acc.count >= frame {
            let f = Array(acc.prefix(frame)); acc.removeFirst(frame)
            handle(f)
        }
    }

    private func updateFloor(_ r: Float) {
        if r < floor { floor = 0.7 * floor + 0.3 * r }
        else if speechFrames == 0 { floor = floor * 1.002 + 0.00002 }
        floor = min(max(floor, 0.0008), 0.03)
    }

    private func handle(_ f: [Float]) {
        let r = f.withUnsafeBufferPointer { rms($0.baseAddress!, f.count) }
        frames += 1
        guard collecting else { updateFloor(r); return }
        if Date().timeIntervalSince(lastLevel) >= 0.06 {
            lastLevel = Date()
            emit(["event": "mic_level", "v": min(1.0, Double(r) * 12), "speech": speechStarted])
        }
        samples.append(contentsOf: f); sinceListen += frame
        if !vad {                                         // push-to-talk: everything until `endpoint`
            if sinceListen >= maxSamples { done = true }
            return
        }
        let thr = max(floor * 3, minThr, minLevel)
        if r > thr { loudSum += r; loudN += 1 }
        if !speechStarted {
            updateFloor(r)
            speechFrames = r > thr ? speechFrames + 1 : 0
            if speechFrames >= 4 {
                speechStarted = true; silenceFrames = 0
                speechStartIdx = max(0, samples.count - frame * 4 - 4000)
            } else if sinceListen * 1000 / 16000 >= noSpeechMs { done = true }
        } else {
            silenceFrames = r < thr ? silenceFrames + 1 : 0
            if silenceFrames * 20 >= silenceEndMs { done = true }
        }
        if sinceListen >= maxSamples { done = true }
    }

    func begin(maxSeconds: Double, waitSeconds: Double, vad: Bool, minLevel: Float = 0) {
        lock.lock()
        self.vad = vad
        self.minLevel = minLevel; loudSum = 0; loudN = 0
        noSpeechMs = Int(waitSeconds * 1000)
        samples = []; collecting = true; speechStarted = !vad; done = false
        speechFrames = 0; silenceFrames = 0; speechStartIdx = 0; sinceListen = 0
        maxSamples = Int(maxSeconds * 16000)
        lock.unlock()
    }
    func forceEnd() { lock.lock(); done = true; lock.unlock() }
    func status() -> (started: Bool, done: Bool) { lock.lock(); defer { lock.unlock() }; return (speechStarted, done) }
    func finish() -> [Float] {
        lock.lock(); defer { lock.unlock() }
        collecting = false
        guard speechStarted else { samples = []; return [] }
        let tail = vad ? max(0, silenceFrames * frame - 3200) : 0       // keep 0.2 s of trailing silence
        let end = max(speechStartIdx, samples.count - tail)
        let out = Array(samples[speechStartIdx..<end])
        samples = []
        return out
    }
}

// MARK: - audio first, then the models
// The engines are started before Core ML loads anything: under memory pressure macOS refuses new audio streams.
let player = Player()
let mic = Mic()
if env["NO_MIC"] != "1" { mic.ensureRunning() }

// Models are read from local folders only. With a model *name* the SDK asks Hugging Face first, which fails offline.
let hf = env["OLINGO_MODELS_DIR"] ?? (NSHomeDirectory() + "/Documents/huggingface/models")
func dir(_ p: String) -> String? {
    var isDir: ObjCBool = false
    return FileManager.default.fileExists(atPath: p, isDirectory: &isDir) && isDir.boolValue ? p : nil
}
let whisperName = env["WHISPER_MODEL"] ?? "large-v3-v20240930_626MB"
let whisperFolder = env["WHISPER_MODEL_PATH"] ?? dir("\(hf)/argmaxinc/whisperkit-coreml/openai_whisper-\(whisperName)")
let ttsFolder = env["TTS_MODELS_PATH"] ?? dir("\(hf)/argmaxinc/ttskit-coreml")
let ttsTok = env["TTS_TOKENIZER_PATH"] ?? dir("\(hf)/Qwen/Qwen3-0.6B")
let speaker = Qwen3Speaker(rawValue: env["TTS_SPEAKER"] ?? "serena") ?? .serena
let ttsVariant: TTSModelVariant = env["TTS_MODEL"] == "1.7b" ? .qwen3TTS_1_7b : .qwen3TTS_0_6b
let decoderMode: Qwen3SpeechDecoderMode = env["TTS_DECODER_MODE"] == "latency" ? .latencyOptimized : .throughputOptimized
let multiMode: Qwen3MultiCodeDecoderMode = env["TTS_MULTICODE"] == "stepped" ? .stepped : .fused
let ttsUnits: MLComputeUnits = { switch env["TTS_UNITS"] ?? "ne" { case "gpu": return .cpuAndGPU; case "cpu": return .cpuOnly; default: return .cpuAndNeuralEngine } }()

func fatal(_ code: String, _ message: String) -> Never {
    log("FATAL \(message)")
    emit(["event": "fatal", "code": code, "message": message])
    exit(2)
}
guard let whisperFolder else { fatal("no_whisper", "Whisper model \(whisperName) is not on this Mac (looked in \(hf)/argmaxinc/whisperkit-coreml)") }
guard let ttsFolder, let ttsTok else { fatal("no_tts", "TTS models or tokenizer are not on this Mac (looked in \(hf))") }

let t0 = Date()
log("loading WhisperKit \(whisperName) from \(whisperFolder)")
let whisper: WhisperKit
do {
    whisper = try await WhisperKit(WhisperKitConfig(
        model: whisperName, modelFolder: whisperFolder,
        computeOptions: ModelComputeOptions(melCompute: .cpuAndGPU, audioEncoderCompute: .cpuAndNeuralEngine,
                                            textDecoderCompute: .cpuAndNeuralEngine),
        verbose: false, prewarm: true, load: true, download: false))
} catch { fatal("whisper_load", "WhisperKit could not load: \(error)") }
log(String(format: "WhisperKit ready in %.1fs", Date().timeIntervalSince(t0)))

// A second, small Whisper for the learner's own language on the open floor: about 0.15 s instead of 0.65 s per
// utterance. When it is unsure, the large model transcribes the same audio again, so accuracy is kept.
let fastName = env["WHISPER_FAST"] ?? "small"
var whisperFast: WhisperKit? = nil
if fastName != "off", fastName != whisperName, let ff = dir("\(hf)/argmaxinc/whisperkit-coreml/openai_whisper-\(fastName)") {
    whisperFast = try? await WhisperKit(WhisperKitConfig(
        model: fastName, modelFolder: ff,
        computeOptions: ModelComputeOptions(melCompute: .cpuAndGPU, audioEncoderCompute: .cpuAndNeuralEngine,
                                            textDecoderCompute: .cpuAndNeuralEngine),
        verbose: false, prewarm: true, load: true, download: false))
    log(whisperFast == nil ? "fast Whisper \(fastName) did not load" : "fast Whisper \(fastName) ready")
}

let t1 = Date()
log("loading TTSKit decoder=\(decoderMode) multi=\(multiMode)")
let tts: TTSKit
do {
    tts = try await TTSKit(TTSKitConfig(
        model: ttsVariant, modelFolder: URL(fileURLWithPath: ttsFolder), tokenizerFolder: URL(fileURLWithPath: ttsTok),
        speechDecoderMode: decoderMode, multiCodeDecoderMode: multiMode,
        computeOptions: ComputeOptions(embedderComputeUnits: .cpuOnly, codeDecoderComputeUnits: ttsUnits,
                                       multiCodeDecoderComputeUnits: ttsUnits, speechDecoderComputeUnits: ttsUnits),
        verbose: false, download: false))
} catch { fatal("tts_load", "TTSKit could not load: \(error)") }
log(String(format: "TTSKit ready in %.1fs", Date().timeIntervalSince(t1)))

// Argmax SpeakerKit: a voice embedding per utterance, to tell the learner's voice from other people's.
let speakerFolder = env["SPEAKER_MODELS_PATH"] ?? dir("\(hf)/argmaxinc/speakerkit-coreml")
var speakerKit: SpeakerKit? = nil
if env["VOICE_ID"] != "off" {
    let ts = Date()
    if env["SPEAKER_DOWNLOAD"] == "1" {                      // one time, online: fetch the models
        speakerKit = try? await SpeakerKit(PyannoteConfig(download: true, load: true, verbose: false))
    } else if let sf = speakerFolder {
        speakerKit = try? await SpeakerKit(PyannoteConfig(modelFolder: sf, download: false, load: true, verbose: false))
    }
    log(speakerKit == nil ? "SpeakerKit not loaded (voice focus off)" : String(format: "SpeakerKit ready in %.1fs", Date().timeIntervalSince(ts)))
}

// Raw voice embeddings all point roughly the same way (what every voice has in common), so their cosine says
// little: measured, another person scores 0.59 to 0.97 against 0.81 to 0.97 for the same person. Subtracting the
// average voice first leaves what is particular to a speaker: same person median 0.64, another person -0.08.
let voiceMean: [Float] = {
    guard let path = env["OLINGO_VOICE_MEAN"], let data = FileManager.default.contents(atPath: path),
          let arr = try? JSONSerialization.jsonObject(with: data) as? [Double] else { return [] }
    return arr.map { Float($0) }
}()
if speakerKit != nil && voiceMean.isEmpty { log("no average-voice file: voice focus off"); speakerKit = nil }
func centred(_ e: [Float]) -> [Float] {
    guard voiceMean.count == e.count else { return e }
    var o = e
    for i in 0..<o.count { o[i] -= voiceMean[i] }
    return o
}

/// The learner's voice: a running mean of utterance embeddings. Lives in this process's memory only.
final class VoiceProfile: @unchecked Sendable {
    private let lock = NSLock()
    private var sum: [Float] = []
    private(set) var seconds = 0.0
    func add(_ raw: [Float], weight: Double) {
        let n = unit(centred(raw))
        lock.lock(); defer { lock.unlock() }
        if sum.count != n.count { sum = [Float](repeating: 0, count: n.count) }
        for i in 0..<n.count { sum[i] += n[i] * Float(weight) }
        seconds += weight
    }
    func similarity(_ raw: [Float]) -> Double? {
        lock.lock(); let s = sum; lock.unlock()
        guard !s.isEmpty, s.count == raw.count else { return nil }
        let a = unit(s), b = unit(centred(raw))
        var d: Float = 0
        for i in 0..<a.count { d += a[i] * b[i] }
        return Double(d)
    }
    func clear() { lock.lock(); sum = []; seconds = 0; lock.unlock() }
    private func unit(_ v: [Float]) -> [Float] {
        var n: Float = 0
        for x in v { n += x * x }
        n = n.squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }
}
let voice = VoiceProfile()

/// The learner's own first answers, kept as audio in memory for the length of the call (never written anywhere).
final class VoiceSample: @unchecked Sendable {
    private let lock = NSLock()
    private var audio: [Float] = []
    func add(_ s: [Float]) { lock.lock(); audio.append(contentsOf: s); if audio.count > 16000 * 9 { audio.removeFirst(audio.count - 16000 * 9) }; lock.unlock() }
    var samples: [Float] { lock.lock(); defer { lock.unlock() }; return audio }
    func clear() { lock.lock(); audio = []; lock.unlock() }
}
let voiceSample = VoiceSample()

/// Is the new utterance spoken by the person in the enrolment audio? Both go into one recording, separated by half a
/// second of silence, and SpeakerKit diarises it: one speaker label across both parts = the same person.
/// -> (same, share): share = how much of the utterance's speech carries the enrolment speaker's label. nil = cannot tell.
func sameSpeaker(_ utterance: [Float]) async -> (same: Bool, share: Double, speakers: Int)? {
    let enrol = voiceSample.samples
    guard let sk = speakerKit, enrol.count >= 16000 * 2, utterance.count >= 12800 else { return nil }
    let gap = [Float](repeating: 0, count: 8000)
    let split = Float(enrol.count + gap.count / 2) / 16000.0
    guard let r = try? await sk.diarize(audioArray: enrol + gap + utterance) else { return nil }
    var before: [Int: Float] = [:], after: [Int: Float] = [:]
    for s in r.segments {
        guard let id = s.speaker.speakerId else { continue }
        let a = max(0, min(s.endTime, split) - s.startTime), b = max(0, s.endTime - max(s.startTime, split))
        if a > 0 { before[id, default: 0] += a }
        if b > 0 { after[id, default: 0] += b }
    }
    guard let me = before.max(by: { $0.value < $1.value })?.key else { return nil }
    let total = after.values.reduce(0, +)
    guard total > 0.3 else { return nil }
    let share = Double((after[me] ?? 0) / total)
    return (share >= 0.5, share, r.speakerCount)
}

/// The embedding of whoever speaks most in these samples. nil when there is too little speech to tell.
func voiceEmbedding(_ samples: [Float]) async -> (emb: [Float], seconds: Double, speakers: Int)? {
    guard let sk = speakerKit, samples.count >= 12800 else { return nil }         // at least 0.8 s
    guard let r = try? await sk.diarize(audioArray: samples) else { return nil }
    var dur: [Int: Float] = [:]
    for s in r.segments { if let id = s.speaker.speakerId { dur[id, default: 0] += s.endTime - s.startTime } }
    guard let best = dur.max(by: { $0.value < $1.value })?.key, let e = r.speakerCentroidEmbeddings[best], !e.isEmpty else { return nil }
    return (e, Double(dur[best] ?? 0), r.speakerCount)
}

let ttsLangs: [String: Qwen3Language] = ["en": .english, "zh": .chinese, "ja": .japanese, "ko": .korean, "de": .german,
                                         "fr": .french, "ru": .russian, "pt": .portuguese, "es": .spanish, "it": .italian]
func ttsLang(_ s: String?) -> Qwen3Language { ttsLangs[s ?? "en"] ?? .english }
/// Seconds of audio per character: only used to decide when streamed playback may start without running dry.
func secPerChar(_ s: String?) -> Double { ["zh", "ja", "ko"].contains(s ?? "en") ? 0.24 : 0.075 }

let asrBase = DecodingOptions(
    task: .transcribe, language: "en", temperatureFallbackCount: 0, sampleLength: envInt("ASR_SAMPLE_LEN", 96),
    usePrefillPrompt: true, detectLanguage: false, wordTimestamps: env["ASR_WORD_TS"] != "0",
    logProbThreshold: nil, firstTokenLogProbThreshold: nil)

// warm-up, so the first real turn is not a cold start
_ = try? await tts.generate(text: "Hello.", speaker: speaker, language: .english)
_ = try? await tts.generate(text: "你好。", speaker: speaker, language: .chinese)
_ = try? await whisper.transcribe(audioArray: [Float](repeating: 0, count: 16000), decodeOptions: asrBase)
_ = try? await whisperFast?.transcribe(audioArray: [Float](repeating: 0, count: 16000), decodeOptions: asrBase)
log(String(format: "bridge warm in %.1fs", Date().timeIntervalSince(t0)))
emit(["event": "ready", "whisper": whisperName, "whisper_fast": whisperFast == nil ? "" : fastName, "tts": ttsVariant == .qwen3TTS_1_7b ? "1.7b" : "0.6b",
      "load_s": Date().timeIntervalSince(t0), "audio_out": player.engine.isRunning, "audio_in": mic.engine.isRunning,
      "voice_id": speakerKit != nil])

// MARK: - speaking
/// How fast speech is generated, in seconds of audio per second (measured, running average).
final class GenRate: @unchecked Sendable {
    private let lock = NSLock(); private var r = 2.0
    var value: Double { lock.lock(); defer { lock.unlock() }; return r }
    func update(_ x: Double) { lock.lock(); r = 0.6 * r + 0.4 * min(max(x, 0.2), 6); lock.unlock() }
}
let genRate = GenRate()

let sayBuffer = Double(env["SAY_BUFFER_S"] ?? "") ?? 0.3      // audio in hand before a sentence starts playing

final class SayState: @unchecked Sendable {
    let lock = NSLock()
    var held: [Float] = [], flushed = false, audioSec = 0.0
    var hp = HighPass()
}

/// The synthesiser's clips carry rumble below 60 Hz and start on a non-zero sample (measured: 14% of the peak in
/// the median clip). Both are heard as a thump at the start of every sentence. A 70 Hz high-pass removes the
/// rumble; the fade-in removes the step. Speech starts above 85 Hz, so nothing audible is lost.
struct HighPass {
    private var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0
    private static let c: (Float, Float, Float, Float, Float) = {
        let w = 2 * Float.pi * 70 / 24000, cs = cos(w), al = sin(w) / (2 * 0.7071)
        let a0 = 1 + al
        return ((1 + cs) / 2 / a0, -(1 + cs) / a0, (1 + cs) / 2 / a0, -2 * cs / a0, (1 - al) / a0)
    }()
    mutating func run(_ x: inout [Float]) {
        let (b0, b1, b2, a1, a2) = HighPass.c
        for i in 0..<x.count {
            let v = x[i]
            let y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1; x1 = v; y2 = y1; y1 = y
            x[i] = y
        }
    }
}
func clean(_ x: [Float]) -> [Float] {                        // a whole cached clip: filter, then fade both ends
    var a = x
    var hp = HighPass()
    hp.run(&a); fadeIn(&a); fadeOut(&a)
    return a
}

func fadeIn(_ x: inout [Float], _ n: Int = 288) {            // 12 ms
    let m = min(n, x.count)
    for i in 0..<m { x[i] *= Float(i) / Float(m) }
}
func fadeOut(_ x: inout [Float], _ n: Int = 288) {
    let m = min(n, x.count)
    for i in 0..<m { x[x.count - 1 - i] *= Float(i) / Float(m) }
}

/// First 10 ms as its own buffer: its completion callback is the moment the item became audible.
func scheduleItem(_ input: [Float], id: String, since: Date) {
    var samples = input
    fadeIn(&samples)
    if let path = env["OLINGO_DUMP_FIRST"] {                 // test hook: the first 0.2 s exactly as it is played
        let n = min(4800, samples.count)
        try? samples[0..<n].map { String($0) }.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }
    let head = min(240, samples.count)
    player.schedule(Array(samples[0..<head])) { emit(["event": "start", "id": id, "ms": ms(since)]) }
    if samples.count > head { player.schedule(Array(samples[head...])) }
}

func doSay(_ d: [String: Any], _ epoch: Int) async {
    let id = d["id"] as? String ?? ""
    let text = (d["text"] as? String) ?? ""
    let lang = d["lang"] as? String
    let mute = d["mute"] as? Bool ?? false               // synthesise + save only (building the phrase cache)
    let start = Date()
    if !mute {
        await player.setRate(Float(d["rate"] as? Double ?? 1.0), epoch)
        if ctl.epoch != epoch { emit(["event": "said", "id": id, "stopped": true]); return }
        player.begin(id)
    }
    let voice = (d["speaker"] as? String).flatMap { Qwen3Speaker(rawValue: $0) } ?? speaker
    let st = SayState()
    let est = Double(text.count) * secPerChar(lang) * 1.2 + 0.3
    var opts = GenerationOptions()
    opts.concurrentWorkerCount = 1
    do {
        let result = try await tts.generate(text: text, speaker: voice, language: ttsLang(lang), options: opts) { p in
            if ctl.epoch != epoch { return false }
            if mute { return true }
            st.lock.lock()
            st.audioSec += Double(p.audio.count) / 24000.0
            var chunk = p.audio
            st.hp.run(&chunk)                            // one filter runs through the whole sentence
            if st.flushed {
                st.lock.unlock()
                if player.lead < 0.004 {                 // playback ran dry in mid-sentence: resume without a jump
                    log("underrun: playback ran dry inside a sentence")
                    fadeIn(&chunk)
                }
                player.schedule(chunk)
                return true
            }
            st.held.append(contentsOf: chunk)
            // Start only when playback cannot run dry. The second chunk of a sentence arrives late (about 0.28 s
            // after the first), so one chunk in hand is not enough: wait for 0.55 s of audio, counting what is still
            // queued ahead (after a filler or the previous sentence that costs no time at all).
            let r = genRate.value
            let have = Double(st.held.count) / 24000.0
            let lead = player.lead
            let go = r >= 1.15 ? (have + lead >= sayBuffer) : (have >= est * (1 - r) + 0.3 - r * lead)
            var h: [Float] = []
            if go { st.flushed = true; h = st.held; st.held = [] }
            st.lock.unlock()
            if go { scheduleItem(h, id: id, since: start) }
            return true
        }
        let elapsed = Date().timeIntervalSince(start)
        if elapsed > 0.05 { genRate.update(result.audioDuration / elapsed) }
        if let save = d["save"] as? String, ctl.epoch == epoch {
            let u = URL(fileURLWithPath: save)
            _ = try? await AudioOutput.saveAudio(result.audio, toFolder: u.deletingLastPathComponent(),
                                                 filename: u.deletingPathExtension().lastPathComponent,
                                                 sampleRate: result.sampleRate, format: .wav)
        }
        if mute {
            emit(["event": "said", "id": id, "ms": ms(start), "audio_s": result.audioDuration, "muted": true,
                  "stopped": ctl.epoch != epoch])
            return
        }
        if ctl.epoch != epoch { player.finish(id, ["stopped": true]); return }
        st.lock.lock()
        let h = st.held; st.held = []; let was = st.flushed; st.flushed = true
        st.lock.unlock()
        if !was { scheduleItem(h, id: id, since: start) }
        let info: [String: Any] = ["audio_s": result.audioDuration, "gen_ms": Int(elapsed * 1000)]
        player.schedule([Float](repeating: 0, count: 240)) {
            var e = info; e["ms"] = ms(start); player.finish(id, e)
        }
        player.watchdog(id)
    } catch {
        if ctl.epoch == epoch { emit(["event": "error", "code": "tts", "message": "say: \(error)"]) }
        if mute { emit(["event": "said", "id": id, "stopped": true]) } else { player.finish(id, ["stopped": true]) }
    }
}

func doPlay(_ d: [String: Any], _ epoch: Int) async {
    let id = d["id"] as? String ?? ""
    let start = Date()
    guard let path = d["file"] as? String, let loaded = player.load(path) else {
        emit(["event": "error", "code": "play", "message": "cannot read \(d["file"] ?? "nil")"])
        emit(["event": "said", "id": id, "stopped": true]); return
    }
    await player.setRate(Float(d["rate"] as? Double ?? 1.0), epoch)
    if ctl.epoch != epoch { emit(["event": "said", "id": id, "stopped": true]); return }
    player.begin(id)
    let samples = clean(loaded)
    let head = min(240, samples.count)
    player.schedule(Array(samples[0..<head])) { emit(["event": "start", "id": id, "ms": ms(start)]) }
    let audio = Double(samples.count) / 24000.0
    player.schedule(samples.count > head ? Array(samples[head...]) : [Float](repeating: 0, count: 120)) {
        player.finish(id, ["ms": ms(start), "audio_s": audio, "cached": true])
    }
    player.watchdog(id)
}

// MARK: - listening
struct Heard { var text = ""; var words: [[String: Any]] = []; var avg = 0.0 }

func transcribe(_ samples: [Float], lang: String, hint: String?, fast: Bool = false) async -> Heard {
    var o = asrBase
    o.language = lang
    let model = fast ? (whisperFast ?? whisper) : whisper
    if env["OLINGO_ASR_HINT"] != "0", let h = hint, !h.isEmpty, let tok = model.tokenizer {
        // The learner is repeating a known word. A single syllable or two, out of context, is where Whisper is at
        // its worst (it writes another word that sounds alike). Telling it the word in play removes most of that.
        o.promptTokens = tok.encode(text: " " + h).filter { $0 < tok.specialTokens.specialTokenBegin }
    }
    // WhisperKit does not transcribe a window shorter than one second (windowClipTime): a single word such as "yes"
    // or a repeated word of two syllables came back empty in a few milliseconds. Short audio gets silence around it.
    var audio = samples
    if audio.count < 24000 {
        audio = [Float](repeating: 0, count: 3200) + audio + [Float](repeating: 0, count: 24000 - audio.count)
    }
    guard let res = try? await model.transcribe(audioArray: audio, decodeOptions: o) else { return Heard() }
    var h = Heard()
    h.text = res.map { $0.text }.joined(separator: " ").trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
    var sum = 0.0
    for w in res.flatMap({ $0.allWords }) {
        let t = w.word.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        if t.isEmpty || t.hasPrefix("<|") { continue }
        h.words.append(["w": t, "p": Double(w.probability)])
        sum += Double(w.probability)
    }
    h.avg = h.words.isEmpty ? 0 : sum / Double(h.words.count)
    return h
}

func doListen(_ d: [String: Any], _ epoch: Int) async {
    func cancelled() { emit(["event": "heard", "text": "", "audio_s": 0.0, "ms": 0, "words": [Any](), "cancelled": true]) }
    let ptt = d["ptt"] as? Bool ?? false
    await player.waitIdle(epoch)                          // never listen to our own voice
    if ctl.epoch != epoch { cancelled(); return }
    if !ptt { try? await Task.sleep(nanoseconds: UInt64(envInt("VAD_GUARD_MS", 250)) * 1_000_000) }
    guard mic.ensureRunning() else {
        emit(["event": "heard", "text": "", "audio_s": 0.0, "ms": 0, "words": [Any](), "error": "audio_input"]); return
    }
    ctl.endpoint = false
    let f0 = mic.framesSeen
    mic.begin(maxSeconds: d["max"] as? Double ?? 12, waitSeconds: d["wait"] as? Double ?? 10, vad: !ptt,
              minLevel: Float(d["min_level"] as? Double ?? 0))
    emit(["event": "listening"])
    var announced = false, warned = false
    let began = Date()
    while ctl.epoch == epoch {
        if ctl.endpoint { mic.forceEnd() }
        let s = mic.status()
        if s.started && !announced { announced = true; emit(["event": "speech_start"]) }
        if s.done { break }
        try? await Task.sleep(nanoseconds: 10_000_000)
        if !warned, Date().timeIntervalSince(began) > 1.2, mic.framesSeen == f0 {
            warned = true
            emit(["event": "error", "code": "audio_input", "message": "the microphone is delivering no audio"])
        }
    }
    let endAt = Date()
    let loud = mic.level
    let samples = mic.finish()
    if ctl.epoch != epoch { cancelled(); return }
    if samples.count < 3200 {                             // nothing, or less than 0.2 s
        emit(["event": "heard", "text": "", "audio_s": 0.0, "ms": 0, "words": [Any](), "timeout": true]); return
    }
    let audioS = Double(samples.count) / 16000.0
    emit(["event": "speech_end", "audio_s": audioS])
    var voiceInfo: [String: Any] = [:]
    if let mode = d["voice"] as? String, speakerKit != nil {
        let tv = Date()
        if let v = await voiceEmbedding(samples) {
            voiceInfo["speech_s"] = v.seconds; voiceInfo["speakers"] = v.speakers
            if mode == "enroll" {
                voice.add(v.emb, weight: max(v.seconds, 0.5))
                voiceSample.add(samples)
            } else if let sim = voice.similarity(v.emb) {
                voiceInfo["sim"] = sim
                // two independent signs must agree before a sentence is dropped: the embedding is unlike the learner's,
                // and SpeakerKit, given both recordings at once, hears two different people
                var other = false
                if sim < (d["voice_min"] as? Double ?? -2), let j = await sameSpeaker(samples) {
                    voiceInfo["share"] = j.share
                    other = !j.same
                }
                if other, v.seconds >= (d["voice_min_s"] as? Double ?? 1.0) {
                    voiceInfo["ms"] = ms(tv)
                    emit(["event": "heard", "text": "", "audio_s": audioS, "ms": ms(endAt), "words": [Any](),
                          "rejected": true, "voice": voiceInfo, "level": loud])
                    return                                // not the learner: no filler, no transcription, no reply
                }
                if sim >= (d["voice_learn"] as? Double ?? 2) { voice.add(v.emb, weight: v.seconds * 0.5) }
            }
        }
        voiceInfo["enrolled_s"] = voice.seconds
        voiceInfo["ms"] = ms(tv)
    }
    // filler: a cached "Hmm." starts the instant the learner stops, while Whisper is still transcribing
    if let f = d["filler"] as? String, let fs = player.load(f) { player.schedule(clean(fs)) }

    var lang = d["lang"] as? String ?? "en"
    var detectMs = 0
    if lang == "auto" {
        let cands = d["langs"] as? [String] ?? ["en", "zh"]
        let td = Date()
        if let r = try? await whisper.detectLangauge(audioArray: samples) {
            lang = cands.max { (r.langProbs[$0] ?? -1e9) < (r.langProbs[$1] ?? -1e9) } ?? cands[0]
        } else { lang = cands[0] }
        detectMs = ms(td)
    }
    let fast = (d["fast"] as? Bool ?? false) && whisperFast != nil
    var h = await transcribe(samples, lang: lang, hint: d["hint"] as? String, fast: fast)
    var tried = [fast ? "\(lang):\(fastName)" : lang]
    if fast && (h.text.isEmpty || h.avg < (Double(env["ASR_FAST_MIN"] ?? "") ?? 0.8)) {   // the small model is unsure
        let hb = await transcribe(samples, lang: lang, hint: d["hint"] as? String)
        tried.append(lang)
        if h.text.isEmpty || hb.avg >= h.avg { h = hb }
    }
    // unsure in the main language: try the other one and keep the more confident transcript
    if let alt = d["alt"] as? String, alt != lang, h.avg < 0.55 {
        let h2 = await transcribe(samples, lang: alt, hint: nil)
        tried.append(alt)
        if h2.avg > h.avg + 0.08 && !h2.text.isEmpty { h = h2; lang = alt }
    }
    if ctl.epoch != epoch { cancelled(); return }
    emit(["event": "heard", "text": h.text, "audio_s": audioS, "ms": ms(endAt), "words": h.words, "lang": lang,
          "avg_p": h.avg, "detect_ms": detectMs, "tried": tried, "level": loud, "voice": voiceInfo])
}

/// Self-test: enrol from, or compare against, a wav file.
func doVoiceTest(_ d: [String: Any]) async {
    guard let path = d["file"] as? String, let buf = try? AudioProcessor.loadAudio(fromPath: path) else {
        emit(["event": "voice", "error": "cannot read file"]); return
    }
    let t = Date()
    guard let v = await voiceEmbedding(AudioProcessor.convertBufferToArray(buffer: buf)) else {
        emit(["event": "voice", "error": "no embedding", "ms": ms(t)]); return
    }
    let samples = AudioProcessor.convertBufferToArray(buffer: buf)
    if d["mode"] as? String == "enroll" { voice.add(v.emb, weight: max(v.seconds, 0.5)); voiceSample.add(samples) }
    let j = d["mode"] as? String == "enroll" ? nil : await sameSpeaker(samples)
    var out: [String: Any] = ["event": "voice", "sim": voice.similarity(v.emb) ?? -9.0, "speech_s": v.seconds,
                              "same": j?.same ?? true, "share": j?.share ?? -1.0,
                              "speakers": v.speakers, "enrolled_s": voice.seconds, "dim": v.emb.count, "ms": ms(t)]
    if d["dump"] as? Bool == true { out["emb"] = v.emb.map { Double($0) } }
    emit(out)
}

func doTranscribe(_ d: [String: Any]) async {
    guard let path = d["file"] as? String, let buf = try? AudioProcessor.loadAudio(fromPath: path) else {
        emit(["event": "transcribed", "text": "", "error": "cannot read file"]); return
    }
    let t = Date()
    let h = await transcribe(AudioProcessor.convertBufferToArray(buffer: buf), lang: d["lang"] as? String ?? "en", hint: nil,
                             fast: d["fast"] as? Bool ?? false)
    emit(["event": "transcribed", "text": h.text, "words": h.words, "avg_p": h.avg, "ms": ms(t)])
}

// MARK: - command loop. `stop`, `endpoint` and `ping` are handled at once on the reader thread; the rest is queued.
let (cmds, cont) = AsyncStream<([String: Any], Int)>.makeStream()
Thread.detachNewThread {
    while let line = readLine() {
        guard let d = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              let c = d["cmd"] as? String else { emit(["event": "error", "code": "json", "message": "bad json"]); continue }
        switch c {
        case "stop": ctl.bump(); player.stopNow()
        case "endpoint": ctl.endpoint = true
        case "mic":                                           // a chunk of the device's microphone
            if let b64 = d["pcm"] as? String, let data = Data(base64Encoded: b64) {
                var out = [Float](repeating: 0, count: data.count / 2)
                data.withUnsafeBytes { raw in
                    let p = raw.bindMemory(to: Int16.self)
                    for i in 0..<out.count { out[i] = Float(Int16(littleEndian: p[i])) / 32768.0 }
                }
                mic.feed(out)
            }
        case "audio_mode":
            let on = d["agent"] as? Bool ?? false
            ctl.bump(); player.stopNow()
            audioMode.agent = on
            audioMode.mic = on && (d["mic"] as? Bool ?? true)
            emit(["event": "audio_mode", "agent": on, "mic": audioMode.mic])
        case "forget_voice": voice.clear(); voiceSample.clear(); emit(["event": "voice_forgotten"])
        case "ping": emit(["event": "pong", "audio_out": player.engine.isRunning, "audio_in": mic.engine.isRunning])
        default: cont.yield((d, ctl.epoch))
        }
    }
    cont.finish()
}

for await (d, epoch) in cmds {
    let c = d["cmd"] as? String ?? ""
    if epoch != ctl.epoch {                               // sent before a `stop`: answer, but do nothing
        switch c {
        case "say", "play": emit(["event": "said", "id": d["id"] as? String ?? "", "stopped": true])
        case "listen": emit(["event": "heard", "text": "", "audio_s": 0.0, "ms": 0, "words": [Any](), "cancelled": true])
        default: break
        }
        continue
    }
    switch c {
    case "say": await doSay(d, epoch)
    case "play": await doPlay(d, epoch)
    case "listen": await doListen(d, epoch)
    case "transcribe": await doTranscribe(d)
    case "voice_test": await doVoiceTest(d)
    case "quit": emit(["event": "bye"]); exit(0)
    default: emit(["event": "error", "code": "cmd", "message": "unknown cmd \(c)"])
    }
}
