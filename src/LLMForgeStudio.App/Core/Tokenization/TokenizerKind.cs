namespace LLMForgeStudio.App.Core.Tokenization;

public enum TokenizerKind
{
    Character,
    Word,
    ByteLevelBpe,
    Unigram,
    WordPiece,
    SimpleBpe,
    HybridFallback,
    HierarchicalExperimental
}
