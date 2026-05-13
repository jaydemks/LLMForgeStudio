namespace LLMForgeStudio.App.Core.Training;

public static class ParameterEstimator
{
    public static long Estimate(ModelConfig c)
    {
        var tokenEmb = (long)c.VocabSize * c.EmbeddingSize;
        var posEmb = (long)c.BlockSize * c.EmbeddingSize;

        // Stima didattica GPT-like:
        // attention: qkv + projection circa 4 * n_embd^2
        // mlp: expand 4x e project back circa 8 * n_embd^2
        // totale blocco circa 12 * n_embd^2 + layernorm/bias minori
        var perBlock = 12L * c.EmbeddingSize * c.EmbeddingSize;
        var blocks = perBlock * c.Layers;
        var lmHead = (long)c.EmbeddingSize * c.VocabSize;
        return tokenEmb + posEmb + blocks + lmHead;
    }

    public static string Human(long value)
    {
        if (value >= 1_000_000_000) return $"{value / 1_000_000_000.0:0.00}B";
        if (value >= 1_000_000) return $"{value / 1_000_000.0:0.00}M";
        if (value >= 1_000) return $"{value / 1_000.0:0.00}K";
        return value.ToString();
    }
}
