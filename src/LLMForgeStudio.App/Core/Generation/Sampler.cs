namespace LLMForgeStudio.App.Core.Generation;

public static class Sampler
{
    public static int Sample(double[] logits, SamplingConfig config)
    {
        if (logits.Length == 0) return 0;
        if (config.Greedy) return ArgMax(logits);

        var temperature = Math.Max(0.05, config.Temperature);
        var scaled = logits.Select(v => v / temperature).ToArray();

        var topK = Math.Clamp(config.TopK, 1, scaled.Length);
        var allowed = scaled
            .Select((v, i) => (Value: v, Index: i))
            .OrderByDescending(x => x.Value)
            .Take(topK)
            .ToArray();

        var max = allowed.Max(x => x.Value);
        var exp = allowed.Select(x => Math.Exp(x.Value - max)).ToArray();
        var sum = exp.Sum();
        var random = new Random(config.Seed);
        var roll = random.NextDouble();
        var acc = 0.0;

        for (var i = 0; i < allowed.Length; i++)
        {
            acc += exp[i] / sum;
            if (roll <= acc) return allowed[i].Index;
        }

        return allowed[^1].Index;
    }

    private static int ArgMax(double[] values)
    {
        var bestIndex = 0;
        var best = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > best)
            {
                best = values[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
