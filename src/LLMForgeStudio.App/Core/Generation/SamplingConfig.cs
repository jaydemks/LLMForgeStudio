namespace LLMForgeStudio.App.Core.Generation;

public sealed class SamplingConfig
{
    public double Temperature { get; set; } = 0.45;
    public int TopK { get; set; } = 30;
    public int Seed { get; set; } = -1;
    public bool Greedy { get; set; } = false;
    public int MaxNewTokens { get; set; } = 80;
}
