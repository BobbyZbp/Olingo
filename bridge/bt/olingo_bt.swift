// olingo-bt: a pipe to a Bluetooth serial (RFCOMM) channel of a paired device, through IOBluetooth.
//   olingo-bt <address, e.g. 98-FE-54-60-3E-7F> [channel, default 1]
// What arrives from the device is written to stdout; what is read from stdin is sent to the device.
// Status lines go to stderr: "open" once the channel is up, "closed" / "error <code>" when it ends.
// Why not /dev/cu.<name>: macOS keeps that port for the service record it saw when the device was paired. After the
// device restarts with a new record the port still opens, but connects to nothing. This asks for the channel directly.
import Foundation
import IOBluetooth

setvbuf(stdout, nil, _IONBF, 0)
func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

var sources: [DispatchSourceSignal] = []

final class Link: NSObject, IOBluetoothRFCOMMChannelDelegate {
    var channel: IOBluetoothRFCOMMChannel?
    let out = FileHandle.standardOutput
    func rfcommChannelOpenComplete(_ ch: IOBluetoothRFCOMMChannel!, status error: IOReturn) {
        if error != kIOReturnSuccess { note("error \(error)"); exit(2) }
        note("open mtu \(ch.getMTU())")
        Thread.detachNewThread { [weak self] in self?.pump() }
    }
    func rfcommChannelData(_ ch: IOBluetoothRFCOMMChannel!, data p: UnsafeMutableRawPointer!, length n: Int) {
        out.write(Data(bytes: p, count: n))
    }
    func rfcommChannelClosed(_ ch: IOBluetoothRFCOMMChannel!) { note("closed"); exit(0) }

    /// stdin -> device, in pieces no larger than the channel's packet size.
    func pump() {
        let inp = FileHandle.standardInput
        while true {
            let d = inp.availableData
            if d.isEmpty { note("stdin closed"); DispatchQueue.main.async { self.channel?.close(); self.channel?.getDevice()?.closeConnection(); exit(0) }; return }
            guard let ch = channel else { return }
            let mtu = max(Int(ch.getMTU()), 64)
            var i = 0
            while i < d.count {
                let n = min(mtu, d.count - i)
                let piece = d.subdata(in: i..<(i + n))
                let r: IOReturn = piece.withUnsafeBytes { raw in
                    ch.writeSync(UnsafeMutableRawPointer(mutating: raw.baseAddress!), length: UInt16(n))
                }
                if r != kIOReturnSuccess { note("write error \(r)"); exit(3) }
                i += n
            }
        }
    }
}

let args = CommandLine.arguments
guard args.count >= 2, let dev = IOBluetoothDevice(addressString: args[1]) else { note("usage: olingo-bt <address> [channel]"); exit(64) }
let chan = BluetoothRFCOMMChannelID(args.count >= 3 ? (UInt8(args[2]) ?? 1) : 1)
let link = Link()

/// Open the channel. A link left over from an earlier run (the program that held it was stopped) makes a new
/// channel wait for ever: if the first try does not come up, the old link is dropped and it is tried once more.
func open(_ attempt: Int) {
    var ch: IOBluetoothRFCOMMChannel?
    let r = dev.openRFCOMMChannelAsync(&ch, withChannelID: chan, delegate: link)
    if r != kIOReturnSuccess && attempt == 2 { note("error \(r)"); exit(2) }
    link.channel = ch
    DispatchQueue.main.asyncAfter(deadline: .now() + (attempt == 1 ? 7 : 10)) {
        if link.channel?.isOpen() ?? false { return }
        if attempt == 2 { note("error timeout"); exit(2) }
        note("no answer: dropping the old link and trying again")
        link.channel?.close()
        dev.closeConnection()
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { open(2) }
    }
}
/// Leave nothing behind: when asked to stop, close the channel and the link first.
for sig in [SIGTERM, SIGINT, SIGHUP] {
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { link.channel?.close(); dev.closeConnection(); exit(0) }
    src.resume()
    sources.append(src)
}
open(1)
RunLoop.main.run()
