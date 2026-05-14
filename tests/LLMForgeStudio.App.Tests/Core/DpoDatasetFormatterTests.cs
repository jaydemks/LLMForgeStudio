using LLMForgeStudio.App.Core.Alignment;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class DpoDatasetFormatterTests
{
    [Fact]
    public void PromptChosenRejectedJsonl_IsFormatted()
    {
        var raw = "{\"prompt\":\"Question\",\"chosen\":\"Good answer\",\"rejected\":\"Bad answer\"}";
        var outText = DpoDatasetFormatter.FormatToTrainingText(raw);

        Assert.Contains("<|user|>", outText);
        Assert.Contains("<|assistant_chosen|>", outText);
        Assert.Contains("<|assistant_rejected|>", outText);
        Assert.Contains("Question", outText);
        Assert.Contains("Good answer", outText);
        Assert.Contains("Bad answer", outText);
    }
}
