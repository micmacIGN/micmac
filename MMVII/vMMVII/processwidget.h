#ifndef PROCESSWIDGET_H
#define PROCESSWIDGET_H

#include <QFrame>
#include <QTimer>
#include <QProcess>
#include <QPushButton>
#include <QVariant>

namespace Ui {
class ProcessWidget;
}

class ProcessWidget : public QFrame
{
    Q_OBJECT

public:
    explicit ProcessWidget(QWidget *parent = nullptr);
    ~ProcessWidget();

    void runCommand(const QString &cmd, const QStringList &args);
    bool isRunning() const;

signals:
    void runningSignal(bool running);

public slots:
    void setSettings();
    void open();

private slots:
    void procStarted();
    void procError(QProcess::ProcessError error);
    void procTimeout();
    void procReadOutput();
    void procReadError();
    void procFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    void addInfo(const QString& msg);
    void addError(const QString& msg);
    void addText(const QString &text);

    void resizeEvent(QResizeEvent *e) override;
    void moveEvent(QMoveEvent *e) override;
    void closeEvent(QCloseEvent *event) override;

    Ui::ProcessWidget *ui;
    QTimer timer;
    QString lastCmd;
    QProcess *proc;
    QPushButton *okButton;
    QPushButton *stopButton;
    QPushButton *clearButton;
    QVariant oldPos;
};


#endif // PROCESSWIDGET_H
