import Darwin
import Foundation
import FoundationModels

@Generable
struct Extraction {
    var passing_trials: Int
    var generated_tokens: Int
}

@main
struct Probe {
    static func main() async {
        do {
            try await extract()
        } catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            exit(1)
        }
    }

    static func extract() async throws {
        let model = SystemLanguageModel.default
        guard case .available = model.availability else {
            throw ProbeError(reason: "On-device model: \(model.availability)")
        }
        let data = FileHandle.standardInput.readDataToEndOfFile()
        let prompt = String(decoding: data, as: UTF8.self)
        let session = LanguageModelSession(model: model)
        let reply = try await session.respond(
            to: prompt, generating: Extraction.self, includeSchemaInPrompt: true,
            options: GenerationOptions(sampling: .greedy, maximumResponseTokens: 128)
        )
        print(reply.rawContent.jsonString)
    }
}

struct ProbeError: Error, CustomStringConvertible {
    let reason: String
    var description: String { reason }
}
