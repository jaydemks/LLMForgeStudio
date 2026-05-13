namespace LLMForgeStudio.App.Core.Generation;

public static class GenerationPreviewService
{
    public static string Describe(SamplingConfig config)
    {
        if (config.Greedy)
            return "Greedy: prende sempre il token con punteggio più alto. Stabile, ma spesso ripetitivo.";

        return $"Sampling: temperature={config.Temperature:0.00}, top_k={config.TopK}, seed={config.Seed}. Più flessibile di greedy.";
    }
}
