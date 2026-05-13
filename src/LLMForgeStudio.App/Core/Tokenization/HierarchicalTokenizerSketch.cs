namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class HierarchicalTokenizerSketch : ITokenizer
{
    public string Name => "Hierarchical experimental sketch";
    public TokenizerKind Kind => TokenizerKind.HierarchicalExperimental;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        // Placeholder intenzionale.
        // Idea futura di Giò: non solo token piatti, ma livelli:
        // macro-token -> subword -> char fallback -> feature/metadati.
        // Attenzione: qui non siamo più nel tokenizer classico. Si entra in tokenizzazione + embedding gerarchico.
        Vocabulary = text.Distinct().OrderBy(c => c).Select((c, i) => new VocabularyItem(i, c.ToString())).ToList();
    }

    public IReadOnlyList<int> Encode(string text) => text.Select(c => Vocabulary.FirstOrDefault(v => v.Token == c.ToString())?.Id ?? 0).ToList();
    public string Decode(IEnumerable<int> ids) => string.Concat(ids.Select(id => Vocabulary.FirstOrDefault(v => v.Id == id)?.Token ?? "?"));
}
