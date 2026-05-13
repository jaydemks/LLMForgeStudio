namespace LLMForgeStudio.App.Core.Training;

public sealed record BatchPreview(IReadOnlyList<int> X, IReadOnlyList<int> Y, string Explanation);
