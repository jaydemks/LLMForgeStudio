namespace LLMForgeStudio.App.Core.Training;

public sealed class CheckpointManifest
{
    public string FormatVersion { get; set; } = "0.1";
    public string ModelWeightsPath { get; set; } = "model.pt";
    public string TokenizerPath { get; set; } = "tokenizer.json";
    public ModelConfig ModelConfig { get; set; } = new();
    public int Step { get; set; }
    public double TrainLoss { get; set; }
    public double ValLoss { get; set; }
}
