namespace LLMForgeStudio.App.Core.Tokenization;

public interface ITokenizer
{
    string Name { get; }
    TokenizerKind Kind { get; }
    IReadOnlyList<VocabularyItem> Vocabulary { get; }

    void Train(string text, TokenizerConfig config);
    IReadOnlyList<int> Encode(string text);
    string Decode(IEnumerable<int> ids);
}
