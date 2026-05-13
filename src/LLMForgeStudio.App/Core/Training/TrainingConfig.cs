namespace LLMForgeStudio.App.Core.Training;

public sealed class TrainingConfig
{
    public int BatchSize { get; set; } = 32;
    public int MaxSteps { get; set; } = 1000;
    public double LearningRate { get; set; } = 3e-4;
    public int EvalEvery { get; set; } = 100;
    public double TrainSplit { get; set; } = 0.9;
    public bool EnableGradientClipping { get; set; } = true;
    public bool ForceCpu { get; set; } = false;
}
