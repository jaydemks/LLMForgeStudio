using System.Windows.Input;

namespace LLMForgeStudio.App.ViewModels;

public sealed class RelayCommand<T> : ICommand
{
    private readonly Action<T?> _execute;
    private readonly Func<T?, bool>? _canExecute;
    public event EventHandler? CanExecuteChanged;

    public RelayCommand(Action<T?> execute, Func<T?, bool>? canExecute = null)
    {
        _execute = execute;
        _canExecute = canExecute;
    }

    public bool CanExecute(object? parameter)
    {
        if (_canExecute is null) return true;
        if (parameter is null) return _canExecute(default);
        if (parameter is T cast) return _canExecute(cast);
        return _canExecute(default);
    }

    public void Execute(object? parameter)
    {
        if (parameter is null)
        {
            _execute(default);
            return;
        }
        if (parameter is T cast)
        {
            _execute(cast);
            return;
        }
        _execute(default);
    }

    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
