# app/medgemma.py

import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class MedGemmaService:
    def __init__(self):
        self.model_id = "google/medgemma-1.5-4b-it"

        print("🏥 Loading MedGemma 1.5 safely...")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # MedGemma is multimodal; pass a dummy image for text-only questions
        self.dummy_image = Image.new("RGB", (896, 896), color="white")

        print(f"✅ Model loaded on {self.model.device}")

    def _extract_final_answer(self, decoded: str) -> str:
        """
        Robustly extract only the final user-facing answer.
        Primary path: capture text between FINAL_ANSWER and END_FINAL_ANSWER.
        Fallback path: aggressively remove planning/rubric text and keep a clean answer.
        """
        decoded = decoded.strip()

        # Remove Gemma <unusedXX> tokens if present
        decoded = re.sub(r"<unused\d+>", "", decoded).strip()

        # 1) PRIMARY: strict marker extraction
        m = re.search(
            r"FINAL_ANSWER:\s*(.*?)\s*END_FINAL_ANSWER",
            decoded,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if m:
            ans = m.group(1).strip()
            # Remove any accidental leftover marker-like text
            ans = re.sub(r"^\s*[-*•]+\s*", "", ans, flags=re.MULTILINE).strip()
            return ans

        # 2) FALLBACK: remove common reasoning/planning sections
        # Drop everything after/including these typical planning headers
        # (We remove lines that contain them, not the entire answer body.)
        bad_patterns = [
            r"^\s*thought\s*$",
            r"^\s*final plan\s*:.*$",
            r"^\s*constraint checklist.*$",
            r"^\s*confidence score.*$",
            r"^\s*mental sandbox.*$",
            r"^\s*attempt\s*\d+.*$",
            r"^\s*define .*:$",
            r"^\s*\(optional\).*$",
            r"^\s*refinement:.*$",
            r"^\s*initial thought:.*$",
            r"^\s*final check.*$",
        ]

        lines = decoded.splitlines()
        cleaned_lines = []
        for line in lines:
            # Skip lines that match any planning/rubric headers
            if any(re.search(p, line, flags=re.IGNORECASE) for p in bad_patterns):
                continue
            # Skip obvious rubric-like bullet instructions
            if re.search(r"\b(do not|avoid|constraint|checklist|plan|simulation)\b", line, flags=re.IGNORECASE):
                # But do NOT skip normal medical sentences that contain "avoid" etc rarely
                # Only skip if it's clearly instruction-y (contains colon or is short)
                if ":" in line or len(line.strip()) < 80:
                    continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()

        # If the fallback still starts with "from the perspective..." style planning, cut it
        # Prefer the first real medical definition sentence patterns.
        anchor = re.search(
            r"\b(Diabetes is a|Diabetes is an|Cancer is a|Cancer is an|Tuberculosis is a|TB is a|Thyroid (disease|problems|disorders)|A thyroid)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if anchor:
            cleaned = cleaned[anchor.start():].strip()

        # Last cleanup: collapse excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        return cleaned

    def generate(self, question: str) -> str:
        # -----------------------------
        # Hard output contract (markers)
        # -----------------------------
        safe_instruction = (
            "You are a medical assistant. Follow these rules строго:\n"
            "1) Do NOT invent medical facts. If unsure, say you are not sure.\n"
            "2) Prefer established medical classifications. Do NOT create new types (e.g., do not say 'Type 3 diabetes').\n"
            "3) Provide concise, patient-friendly explanations.\n"
            "Do NOT explain your reasoning, steps, plans, checks, or thought process.\n"
            "4) Do NOT give prescriptions, dosing, or treatment plans.\n"
            "Do NOT say that you are not qualified or that you cannot provide information.\n"
            "When defining a disease, do NOT include symptoms unless the user explicitly asks for symptoms.\n"
            "You MAY list common symptoms ONLY when the user explicitly asks for symptoms, in a general, educational, non-diagnostic manner.\n"
            "5) End with: 'Consult a healthcare professional for personalized advice.'\n\n"
            f"User question: {question}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.dummy_image},
                    {"type": "text", "text": safe_instruction},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # -----------------------------
        # Deterministic, anti-loop generation
        # -----------------------------
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )

        decoded = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        final_answer = self._extract_final_answer(decoded)

        # Fallback if extraction fails or returns empty
        if not final_answer:
            return (
                "I’m unable to provide an answer to that question right now. "
                "Please consult a healthcare professional for reliable medical guidance."
            )

        return final_answer
