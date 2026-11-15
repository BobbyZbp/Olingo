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
            if d.isEmpty { note("stdin closed"); DispatchQueue.main.async { self.channel?.close(); exit(0) }; return }
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
var ch: IOBluetoothRFCOMMChannel?
let r = dev.openRFCOMMChannelAsync(&ch, withChannelID: chan, delegate: link)
if r != kIOReturnSuccess { note("error \(r)"); exit(2) }
link.channel = ch
DispatchQueue.main.asyncAfter(deadline: .now() + 15) { if !(link.channel?.isOpen() ?? false) { note("error timeout"); exit(2) } }
RunLoop.main.run()
