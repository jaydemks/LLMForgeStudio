using LLMForgeStudio.App.Core.Alignment;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class RlhfFeedbackCollectorTests
{
    [Fact]
    public async Task SaveJsonlAsync_WritesPromptChosenRejected()
    {
        var root = Path.Combine(Path.GetTempPath(), $"llmforge-rlhf-collector-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            var path = Path.Combine(root, "feedback.jsonl");
            var items = new[]
            {
                new RlhfFeedbackRecord { Prompt = "p1", Chosen = "c1", Rejected = "r1" },
                new RlhfFeedbackRecord { Prompt = "p2", Chosen = "c2", Rejected = string.Empty }
            };

            await RlhfFeedbackCollector.SaveJsonlAsync(items, path);
            Assert.True(File.Exists(path));
            var text = await File.ReadAllTextAsync(path);
            Assert.Contains("\"prompt\":\"p1\"", text);
            Assert.Contains("\"chosen\":\"c2\"", text);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
