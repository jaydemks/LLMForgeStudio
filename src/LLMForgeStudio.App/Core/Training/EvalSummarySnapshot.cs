namespace LLMForgeStudio.App.Core.Training;

public sealed class EvalSummarySnapshot
{
    public string EvalSuite { get; set; } = "basic";
    public int NumBenchmarks { get; set; }
    public double AverageScore { get; set; }
    public string Band { get; set; } = "unknown";
    public bool ReleaseGatePassed { get; set; }
    public double ReleaseGateThreshold { get; set; }
}
