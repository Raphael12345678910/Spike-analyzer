import OpenAI from "openai";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: "Missing OPENAI_API_KEY environment variable."
      });
    }

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

Use the provided measurements as evidence, but do not pretend they are perfect.
Do not invent exact ball-contact timing because the app does not track the ball yet.
If a metric seems weak or limited, say so directly.

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

Player analysis data:

- Hitting hand: ${hittingHand ?? "unknown"}
- Camera angle: ${cameraAngle ?? "unknown"}
- Rep type: ${repType ?? "unknown"}
- User notes: ${userNotes ?? "none"}

- Elbow angle near likely contact: ${elbowAngle ?? "unknown"}
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
`;

    const response = await client.responses.create({
      model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
      input: prompt
    });

    return res.status(200).json({
      feedback: response.output_text || "No coaching feedback returned."
    });
  } catch (error) {
    console.error("Coach API error:", error);
    return res.status(500).json({
      error: "Failed to generate coaching feedback."
    });
  }
}
