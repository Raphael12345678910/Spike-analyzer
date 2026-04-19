import OpenAI from "openai";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const {
      hittingHand,
      cameraAngle,
      repType,
      userNotes,
      elbowAngle,
      extensionScore,
      reachEfficiencyScore,
      estimatedContactReach,
      gainAboveStandingReach,
      marginAboveNet,
      warnings
    } = req.body || {};

    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    const prompt = `
You are a clear, practical volleyball coach.

Use the provided measurements as the main evidence.
Do not invent exact timing or exact ball-contact facts if the app did not measure them.
Do not overstate confidence.
Be specific, concise, and useful.

Return exactly this format:

Biggest issue:
<1 short paragraph>

Why it matters:
<1 short paragraph>

What to change:
- <cue 1>
- <cue 2>
- <cue 3>

One drill:
<1 short paragraph>

Confidence / limitations:
<1 short paragraph>

Here is the player's analysis data:

- Hitting hand: ${hittingHand ?? "unknown"}
- Camera angle: ${cameraAngle ?? "unknown"}
- Rep type: ${repType ?? "unknown"}
- User notes: ${userNotes ?? "none"}

- Elbow angle at likely contact: ${elbowAngle ?? "unknown"}
- Extension score: ${extensionScore ?? "unknown"}/10
- Reach efficiency score: ${reachEfficiencyScore ?? "unknown"}/10

- Estimated contact reach: ${estimatedContactReach ?? "not available"}
- Estimated gain above standing reach: ${gainAboveStandingReach ?? "not available"}
- Estimated margin above net: ${marginAboveNet ?? "not available"}

- Warnings: ${
      Array.isArray(warnings) && warnings.length > 0
        ? warnings.join(", ")
        : "none"
    }

Prioritize the single biggest issue first.
If measurement quality is limited, say that clearly.
`;

    const response = await client.responses.create({
      model: "gpt-5.4-mini",
      input: prompt
    });

    const text = response.output_text || "No coaching feedback returned.";
    return res.status(200).json({ feedback: text });
  } catch (error) {
    console.error(error);
    return res.status(500).json({
      error: "Failed to generate coaching feedback."
    });
  }
}