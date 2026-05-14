using LLMForgeStudio.App.Core.Alignment;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class SftDatasetFormatterTests
{
    [Fact]
    public void PromptResponseJsonl_IsFormatted()
    {
        var raw = "{\"prompt\":\"Hi\",\"response\":\"Hello\"}";
        var outText = SftDatasetFormatter.FormatToTrainingText(raw);

        Assert.Contains("<|user|>", outText);
        Assert.Contains("Hi", outText);
        Assert.Contains("<|assistant|>", outText);
        Assert.Contains("Hello", outText);
    }

    [Fact]
    public void MessagesJsonl_IsFormatted()
    {
        var raw = "{\"messages\":[{\"role\":\"user\",\"content\":\"A\"},{\"role\":\"assistant\",\"content\":\"B\"}]}";
        var outText = SftDatasetFormatter.FormatToTrainingText(raw);

        Assert.Contains("<|user|>", outText);
        Assert.Contains("<|assistant|>", outText);
        Assert.Contains("A", outText);
        Assert.Contains("B", outText);
    }
}
