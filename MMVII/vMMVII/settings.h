#ifndef SETTINGS_H
#define SETTINGS_H

#include <QDialog>

namespace Ui {
class Settings;
}

class Settings : public QDialog
{
    Q_OBJECT

public:
    explicit Settings(QWidget *parent = nullptr);
    ~Settings();

    static void setWorkingDirs(const QStringList &dirs);
    static QStringList workingDirs();
    static int maxCommandHistory();
    static bool mmviiWindows();
    static int maxOutputLines();
    static int outputFontSize();
    static QSize outputSize();
    static void setOutputSize(const QSize& size);

signals:
    void workingDirsClearedSignal();
    void maxCommandHistoryChanged();
    void maxOutputLinesChanged();
    void outputFontSizeChanged();

private:
    void accept() override;

private:
    void clearWorkingDirs();
    static void setMaxCommandHistory(int max);
    static void setMmviiWindows(bool on);
    static void setMaxOutputLines(int lines);
    static void setOutputFontSize(int size);
    void saveSettings();

    Ui::Settings *ui;

};

#endif // SETTINGS_H
