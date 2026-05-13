namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class HybridFallbackTokenizer : ITokenizer
{
    private readonly SimpleBpeTokenizer _bpe = new();
    private readonly CharacterTokenizer _chars = new();
    private int _bpeVocabCount;

    public string Name => "Hybrid fallback";
    public TokenizerKind Kind => TokenizerKind.HybridFallback;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        // Ibrido: prova BPE prima; token sconosciuti passano al canale character.
        _bpe.Train(text, config);
        _chars.Train(text, config);
        _bpeVocabCount = _bpe.Vocabulary.Count;

        Vocabulary = _bpe.Vocabulary
            .Concat(_chars.Vocabulary.Select(v => new VocabularyItem(v.Id + _bpeVocabCount, $"char:{v.Token}", v.Frequency)))
            .ToList();
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var bpeIds = _bpe.Encode(text);
        if (!bpeIds.Contains(0)) return bpeIds;

        var result = new List<int>(bpeIds.Count);
        var charIds = _chars.Encode(text);
        var charIdx = 0;

        foreach (var id in bpeIds)
        {
            if (id != 0)
            {
                result.Add(id);
                continue;
            }

            // ID 0 in BPE è <unk>: fallback a carattere per non perdere informazione.
            if (charIdx < charIds.Count)
            {
                result.Add(_bpeVocabCount + charIds[charIdx]);
                charIdx++;
            }
            else
            {
                result.Add(id);
            }
        }

        return result;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var list = ids.ToList();
        var bpePart = new List<int>();
        var charPart = new List<int>();
        foreach (var id in list)
        {
            if (id >= _bpeVocabCount) charPart.Add(id - _bpeVocabCount);
            else bpePart.Add(id);
        }

        if (charPart.Count == 0) return _bpe.Decode(bpePart);
        return _bpe.Decode(bpePart) + _chars.Decode(charPart);
    }
}
