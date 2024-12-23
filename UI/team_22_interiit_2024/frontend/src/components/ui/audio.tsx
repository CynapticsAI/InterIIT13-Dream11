import React, { useState } from "react";
import axios from "axios";
import { cva } from "class-variance-authority";

const buttonStyles = cva(
    "w-5 h-5 absolute -right-[0.5rem] -bottom-[0.5rem] rounded-md p-1 bg-gradient-to-br from-pink-400 to-pink-600 transition-opacity duration-300",
    {
      variants: {
        loading: {
          true: "bg-gray-400 cursor-not-allowed",
          false: "hover:scale-105 hover:bg-gray-400 border-pink-700",
        },
      },
    }
  );

export default function AudioBar({ text }: { text: string }) {
  const [loading, setLoading] = useState(false);

  const textToSpeech = async (text: string) => {
    const XI_API_KEY = "sk_5c32e7bfaf1c8f4b75a63e792189cbc83e23f6d3ff697663";
    const voiceId = "XB0fDUnXU5powFXDhCwa";
    const ttsUrl = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream`;
    const headers = {
      "xi-api-key": XI_API_KEY,
      "Accept": "application/json",
    };

    const data = {
      text,
      model_id: "eleven_multilingual_v2",
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.8,
        style: 0.0,
        use_speaker_boost: true,
      },
    };

    try {
      const response = await axios.post(ttsUrl, data, {
        headers,
        responseType: "arraybuffer",
      });

      const audioBlob = new Blob([response.data], { type: "audio/mpeg" });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
    } catch (error) {
      // @ts-ignore
      throw new Error(`Error generating TTS: ${error.message}`);
    }
  };

  const handleClick = async () => {
    if (loading) return;
    setLoading(true);
    try {
      await textToSpeech(text);
    } catch (error) {
      console.error("Error generating TTS:", error);
    } finally {
      setLoading(false); 
    }
  };

  return (
    <div
        className={`${buttonStyles({ loading })} flex justify-center items-center`}
        onClick={handleClick}
        >
        {!loading ? (
            <img src="/speaker-wave-2-svgrepo-com.svg" className="invert"/>
        ) : (
            <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="animate-spin text-white w-4 h-4"
            >
            <path
                fillRule="evenodd"
                d="M10 18a8 8 0 1 1 0-16 8 8 0 0 1 0 16zm0-2A6 6 0 1 0 10 4a6 6 0 0 0 0 12z"
                clipRule="evenodd"
            />
            </svg>
        )}
    </div>
  );
}
