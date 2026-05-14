namespace LLMForgeStudio.App.Core.Alignment;

public sealed class RlhfFeedbackRecord
{
    public string Prompt { get; set; } = string.Empty;
    public string Chosen { get; set; } = string.Empty;
    public string Rejected { get; set; } = string.Empty;
}
