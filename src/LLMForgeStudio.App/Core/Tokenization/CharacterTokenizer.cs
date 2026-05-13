namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class CharacterTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _stoi = new();
    private readonly Dictionary<int, char> _itos = new();

    public string Name => "Character-level";
    public TokenizerKind Kind => TokenizerKind.Character;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        if (config.LowercaseBeforeTraining) text = text.ToLowerInvariant();

        // Qui c'è la versione leggibile di: { c: i for i, c in enumerate(chars) }.
        // chars = tutti i caratteri unici nel dataset.
        // i = posizione/ID del carattere nel vocabolario.
        // c = carattere corrente.
        // stoi = string/char -> integer. Cioè: 'a' -> 0, 'b' -> 1...
        // itos = integer -> string/char. Cioè: 0 -> 'a', 1 -> 'b'...
        var chars = text.Distinct().OrderBy(c => c).ToArray();
        _stoi.Clear();
        _itos.Clear();

        for (var i = 0; i < chars.Length; i++)
        {
            var c = chars[i];
            _stoi[c] = i;
            _itos[i] = c;
        }

        Vocabulary = chars.Select((c, i) => new VocabularyItem(i, Printable(c), text.Count(x => x == c))).ToList();
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var result = new List<int>(text.Length);
        foreach (var c in text)
        {
            if (_stoi.TryGetValue(c, out var id))
                result.Add(id);
        }
        return result;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var chars = ids.Select(id => _itos.TryGetValue(id, out var c) ? c : '?');
        return new string(chars.ToArray());
    }

    private static string Printable(char c) => c switch
    {
        '\n' => "\\n",
        '\t' => "\\t",
        ' ' => "[space]",
        _ => c.ToString()
    };
}
