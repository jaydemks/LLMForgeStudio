namespace LLMForgeStudio.App.Core.Hardware;

public sealed class HardwareSummary
{
    public string CpuName { get; set; } = "Unknown CPU";
    public int LogicalCores { get; set; }
    public double TotalRamGb { get; set; }
    public IReadOnlyList<string> Gpus { get; set; } = Array.Empty<string>();
    public string OsDescription { get; set; } = string.Empty;
    public string BackendDeviceStatus { get; set; } = "Backend non verificato";
    public string BackendNotes { get; set; } = string.Empty;
}
