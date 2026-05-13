namespace LLMForgeStudio.App.Core.Tokenization;

public static class TokenizerRegistry
{
    public static IReadOnlyList<TokenizerKind> Kinds { get; } = Enum.GetValues<TokenizerKind>();

    public static ITokenizer Create(TokenizerKind kind) => kind switch
    {
        TokenizerKind.Character => new CharacterTokenizer(),
        TokenizerKind.Word => new WordTokenizer(),
        TokenizerKind.SimpleBpe => new SimpleBpeTokenizer(),
        TokenizerKind.HybridFallback => new HybridFallbackTokenizer(),
        TokenizerKind.HierarchicalExperimental => new HierarchicalTokenizerSketch(),
        _ => new CharacterTokenizer()
    };

    public static string Explain(TokenizerKind kind) => kind switch
    {
        TokenizerKind.Character => "Ogni carattere diventa un token. Perfetto per imparare, poco efficiente per modelli seri.",
        TokenizerKind.Word => "Ogni parola/simbolo diventa un token. Intuitivo, ma fragile con parole nuove.",
        TokenizerKind.SimpleBpe => "Parte da caratteri e fonde coppie frequenti. Più realistico e più efficiente.",
        TokenizerKind.HybridFallback => "Usa subword, ma mantiene fallback a caratteri. Buona base per esperimenti pratici.",
        TokenizerKind.HierarchicalExperimental => "Laboratorio per token multilivello: macro/subword/char/metadati. Non produzione.",
        _ => string.Empty
    };
}
