#ifndef ACTIONBOX_H
#define ACTIONBOX_H

#include <QWidget>
#include <QStatusBar>

namespace Ui {
class ActionBox;
}

class ActionBox : public QWidget
{
    Q_OBJECT

public:
    explicit ActionBox(QWidget *parent = nullptr);
    ~ActionBox();

    void setCommandSelection(bool commandSelection);
    void setStatusMessage(const QString& msg);

public slots:
    void runEnabled(bool on);
    void editEnabled(bool on);

signals:
    void quitSignal();
    void settingsSignal();
    void backSignal();
    void editSignal();
    void clearSignal();
    void runEditedSignal();
    void runSelectedSignal();

private:
    Ui::ActionBox *ui;

    QStatusBar *statusBar;
};

#endif // ACTIONBOX_H
