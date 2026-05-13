namespace LLMForgeStudio.App.Core.Training;

public static class DryRunTrainer
{
    public static IEnumerable<TrainingLogEntry> Simulate(TrainingConfig config, int tokenCount)
    {
        var random = new Random(42);
        var loss = 4.2;
        for (var step = 0; step <= config.MaxSteps; step += Math.Max(1, config.EvalEvery))
        {
            loss = Math.Max(1.1, loss * 0.92 + random.NextDouble() * 0.05);
            var val = loss + 0.1 + random.NextDouble() * 0.2;
            yield return new TrainingLogEntry(step, loss, val, tokenCount / Math.Max(1.0, config.BatchSize), step == 0 ? "Dry-run started" : "Dry-run update");
        }
    }
}
