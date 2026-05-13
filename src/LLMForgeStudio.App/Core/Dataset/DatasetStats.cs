namespace LLMForgeStudio.App.Core.Dataset;

public sealed record DatasetStats(
    int CharacterCount,
    int LineCount,
    int ApproxWordCount,
    int UniqueCharacterCount,
    string Preview);
