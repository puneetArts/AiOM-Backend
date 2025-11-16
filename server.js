import 'dotenv/config';
import express from "express";
import cors from "cors";
import { vectorStore } from "./prepare.js";
import Groq from "groq-sdk";

const app = express();
// CORS 
const allowedOrigins = [
  "http://localhost:5173",      // Dev 
  "http://localhost:3000",      //def server
  "https://om-electricals.vercel.app/"  //  deployment
];

app.use(
  cors({
    origin: function (origin, callback) {
      if (!origin) return callback(null, true);

      if (allowedOrigins.includes(origin)) {
        return callback(null, true);
      } else {
        return callback(new Error("Not allowed by CORS"));
      }
    },
    credentials: true,
  })
);
app.use(express.json());

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;

    // 1. Retrieve relevant chunks
    const relevantChunks = await vectorStore.similaritySearch(message, 3);

    const context = relevantChunks
      .map((chunk) => chunk.pageContent)
      .join("\n\n");

    const SYSTEM_PROMPT = `You are AiOM-chatBot, an intelligent, context-aware assistant designed for fast and accurate question-answering tasks for our company.

Your responsibilities:

Use the retrieved context to answer user questions precisely.

If the answer is not available, respond politely without guessing. 
Example fallback:
"I'm not able to find this exact information at the moment. Is there anything else I can help you with?"

Keep answers clear, helpful, and concise.

Don't use **abc** to bold or highlight a text or heading. 

Formatting rules:
- Preserve all line breaks and bullet points exactly.
- If the context contains lists, output them in clean multiline format.
- Do NOT merge items into a single line.
- Keep the response concise, clear, and professional.


Do not invent or assume information that is not in the provided context.

When relevant, synthesize and summarize the information from context in a professional tone.

If the user asks a question that is not answerable using the context, respond with "I don't know."

Your goal is to deliver accurate, context-based, trustworthy responses every time.`;

    const userQuery = `Question: ${message}
Relevant context:
${context}
Answer:`;

    // 2. Call Groq
    const completion = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userQuery }
      ]
    });

    const answer = completion.choices[0].message.content;

    // 3. Send to frontend
    res.json({ message: answer });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Something went wrong." });
  }
});

app.listen(3001, () => {
  console.log("🚀 API server running at http://localhost:3001");
});
