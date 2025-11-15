import readline from "node:readline/promises";
import Groq from "groq-sdk";
import { vectorStore } from "./prepare.js";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY }); // 1

export async function chat() {
  //2
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  }); //3

  while (true) {
    const question = await rl.question("You: "); //4

    if (question === "/bye") {
      break;
    }
    //RETRIEVAL
    const relevantChunks = await vectorStore.similaritySearch(question, 3); //5

    const context = relevantChunks
      .map((chunk) => chunk.pageContent)
      .join("\n\n");

    const SYSTEM_PROMPT = `'You are an assistant for question answering tasks. Use the following relevant pieces of retrieved context to answer the question. If you don't know the answer say i don't know.`;
    

    const userQuery= `Question: ${question}
    Relevant context: ${context}
    Answer:`;
    const completion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: 'system',
          content: SYSTEM_PROMPT,
        },
        {
          role: "user",
          content: userQuery,
        },
      ],
    });

    console.log(`Assistant: ${completion.choices[0].message.content}`);
    
  }
  rl.close();
}
chat();
