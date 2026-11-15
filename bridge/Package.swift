// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "olingo-bridge",
    platforms: [.macOS(.v15)],
    dependencies: [.package(path: "../../argmax-oss-swift")],
    targets: [
        .executableTarget(
            name: "OlingoBridge",
            dependencies: [
                .product(name: "WhisperKit", package: "argmax-oss-swift"),
                .product(name: "TTSKit", package: "argmax-oss-swift"),
                .product(name: "SpeakerKit", package: "argmax-oss-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
