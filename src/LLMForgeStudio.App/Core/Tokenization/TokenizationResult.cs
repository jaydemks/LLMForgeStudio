namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class TokenizationResult
{
    public required IReadOnlyList<int> TokenIds { get; init; }
    public required IReadOnlyList<VocabularyItem> Vocabulary { get; init; }
    public required string DecodedPreview { get; init; }
    public int VocabSize => Vocabulary.Count;
    public int TokenCount => TokenIds.Count;
}
