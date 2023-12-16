import { fileURLToPath } from "url";
import path from "path";
import { LlamaModel, LlamaContext, LlamaChatSession } from "node-llama-cpp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const runLlamaChat = async () => {
    const model = new LlamaModel({
        modelPath: path.join(__dirname, "models", "llama-2-7b-chat.ggmlv3.q8_0.bin")
    });
    const context = new LlamaContext({ model });
    const session = new LlamaChatSession({ context });

    const q1 = "Hi there, how are you?";
    console.log("User: " + q1);

    const a1 = await session.prompt(q1);
    console.log("AI: " + a1);

    const q2 = "Summarize what you said";
    console.log("User: " + q2);

    const a2 = await session.prompt(q2);
    console.log("AI: " + a2);
};

// Call the asynchronous function
runLlamaChat();
