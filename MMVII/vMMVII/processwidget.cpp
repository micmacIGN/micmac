#include "processwidget.h"
#include "ui_processwidget.h"

#include "global.h"
#include "signal.h"
#include "settings.h"

#include <QProcess>
#include <QPushButton>
#include <QLabel>
#include <QDir>
#include <QScrollBar>


ProcessWidget::ProcessWidget(QWidget *parent) :
    QFrame(parent),
    ui(new Ui::ProcessWidget),
    proc(nullptr)
{
    ui->setupUi(this);

    setSettings();
    timer.setSingleShot(true);
    timer.callOnTimeout(this, &ProcessWidget::procTimeout);
    ui->textOutput->setWordWrapMode(QTextOption::WrapAnywhere);
    okButton = ui->buttonBox->button(QDialogButtonBox::Ok);
    okButton->setDefault(true);

    stopButton = ui->buttonBox->button(QDialogButtonBox::Cancel);
    stopButton->setText(tr("&Stop"));
    ui->buttonBox->addButton(stopButton,QDialogButtonBox::AcceptRole);

    clearButton = ui->buttonBox->button(QDialogButtonBox::Reset);
    clearButton->setText(tr("&Clear"));
    clearButton->setIcon(QIcon());

    setWindowTitle(tr("%1 execution").arg(MMVII_EXE_FILE));

    connect(clearButton,&QPushButton::clicked,ui->textOutput,&QPlainTextEdit::clear);
    connect(clearButton,&QPushButton::clicked,clearButton,&QPushButton::setEnabled);
    connect(okButton,&QPushButton::clicked,this,&QFrame::close);

    clearButton->setDisabled(true);
    okButton->setEnabled(true);
    stopButton->setEnabled(false);
}

ProcessWidget::~ProcessWidget()
{
    delete ui;
    delete proc;
}


void ProcessWidget::setSettings()
{
    ui->textOutput->setMaximumBlockCount(Settings::maxOutputLines());
    QFont font = ui->textOutput->font();
    font.setPointSize(Settings::outputFontSize());
    ui->textOutput->setFont(font);
}


void ProcessWidget::open()
{
    QSize size = Settings::outputSize();
    if (size.isValid() && size.width() < 4000 && size.height() < 4000)
        resize(size);
    else
        resize(ui->textOutput->fontMetrics().maxWidth() * 90 + 20, ui->textOutput->fontMetrics().height() * 30 + 20 + ui->buttonBox->height());
    if (oldPos.isValid())
        move(oldPos.toPoint());
    show();
    raise();
}

void ProcessWidget::runCommand(const QString& cmd, const QStringList& args)
{
    okButton->setEnabled(false);
    stopButton->setEnabled(false);
    clearButton->setEnabled(false);

    ui->textOutput->textCursor().clearSelection();
    QTextCursor tc = ui->textOutput->textCursor();
    tc.movePosition(QTextCursor::End);
    ui->textOutput->setTextCursor(tc);
    addInfo("===========================================================");
    addInfo(tr("Starting %1.").arg(MMVII_EXE_FILE));
    QTextCharFormat tf = ui->textOutput->currentCharFormat();
    tf.setForeground(Qt::black);
    ui->textOutput->setCurrentCharFormat(tf);
    ui->textOutput->appendPlainText("");
    ui->textOutput->verticalScrollBar()->setValue(ui->textOutput->verticalScrollBar()->maximum());

    lastCmd = cmd;
    QStringList cmdArgs = args;
    cmdArgs.append(MMVII_VMMVII_SPEC_ARG);

    delete proc;
    proc = new QProcess();
    connect(proc, &QProcess::started, this, &ProcessWidget::procStarted);
    connect(proc, &QProcess::errorOccurred, this, &ProcessWidget::procError);
    connect(proc, &QProcess::readyReadStandardOutput, this, &ProcessWidget::procReadOutput);
    connect(proc, &QProcess::readyReadStandardError, this, &ProcessWidget::procReadError);
    connect(proc, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, &ProcessWidget::procFinished);
    connect(stopButton, &QPushButton::clicked, proc, &QProcess::kill);

    timer.start(3000);
    if (Settings::mmviiWindows()) {
        proc->setProcessChannelMode(QProcess::SeparateChannels);
        proc->start(lastCmd,cmdArgs, QIODevice::ReadOnly | QIODevice::Text);
    } else {
        proc->setProcessChannelMode(QProcess::ForwardedChannels);
        proc->start(lastCmd,cmdArgs, QIODevice::NotOpen);
    }
    emit runningSignal(true);
    open();
}

bool ProcessWidget::isRunning() const
{
    return proc != nullptr && proc->state() != QProcess::NotRunning;
}


void ProcessWidget::resizeEvent(QResizeEvent *e)
{
    Settings::setOutputSize(e->size());
}

void ProcessWidget::moveEvent(QMoveEvent *e)
{
    oldPos = e->pos();
}

void ProcessWidget::closeEvent(QCloseEvent *event)
{
    if (isRunning()) {
        event->ignore();
        return;
    }
    QFrame::closeEvent(event);
}

void ProcessWidget::procStarted()
{
    timer.stop();
    stopButton->setEnabled(true);
    if (! Settings::mmviiWindows())
        addText(tr("\n%1 output goes to the console.").arg(MMVII_EXE_FILE));
}

void ProcessWidget::procFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    ui->textOutput->verticalScrollBar()->setValue(ui->textOutput->verticalScrollBar()->maximum());
    stopButton->setEnabled(false);
    okButton->setEnabled(true);
    clearButton->setEnabled(true);
    switch (exitStatus) {
    case QProcess::NormalExit:
        addText("\n");
        addInfo(tr("%1 terminated with code %2").arg(MMVII_EXE_FILE).arg(exitCode));
        break;
    case QProcess::CrashExit:
#if !defined WIN32 && !defined __WIN32__
        switch (exitCode) {
        case SIGABRT:
            addError(tr("%1 exited with an error.").arg(MMVII_EXE_FILE));
            break;
        case SIGKILL:
            addError(tr("%1 was stopped.").arg(MMVII_EXE_FILE));
            break;
        default:
            addError(tr("%1 crashed with code %2").arg(MMVII_EXE_FILE).arg(exitCode));
            break;
        }
#else
        addError(tr("%1 terminated unexpectedly").arg(MMVII_EXE_FILE));
#endif
        break;
    }
    delete proc;
    proc = nullptr;
    emit runningSignal(false);
}

void ProcessWidget::procError(QProcess::ProcessError error)
{
    switch (error) {
    case QProcess::FailedToStart:
        addText("\n");
        addError(tr("Fatal error : Can't run '%1'").arg(lastCmd));
        stopButton->setEnabled(false);
        okButton->setEnabled(true);
        clearButton->setEnabled(true);
        break;
    case QProcess::Crashed:         // will be handled in procFinished with exitStatus = CrashExit
        break;
    case QProcess::Timedout:        // Should not happed, we don't use waitForXXX()
        break;
    case QProcess::WriteError:
    case QProcess::ReadError:
        addError(tr("Error: Cannot capture command output."));
        break;
    case QProcess::UnknownError:    // Should not happen ....
        break;

    }
}


void ProcessWidget::procTimeout()
{
    addError(tr("Fatal error : timeout occcured while trying to start '%1'").arg(lastCmd));
    if (proc)
        proc->kill();
    stopButton->setEnabled(false);
    clearButton->setEnabled(true);
    okButton->setEnabled(true);
    emit runningSignal(false);
}

void ProcessWidget::procReadOutput()
{
    addText(proc->readAllStandardOutput());
}

void ProcessWidget::procReadError()
{
    addText(proc->readAllStandardError());
}


void ProcessWidget::addText(const QString &text)
{
    QString t=text;
    if (t.endsWith("\n"))
        t.chop(1);
    ui->textOutput->appendPlainText(t);
}


void ProcessWidget::addInfo(const QString &msg)
{
    ui->textOutput->appendHtml(QString("<p style=\"color:%1\">").arg(OUTPUT_CONSOLE_INFO_COLOR) + msg + "</p>");
    ui->textOutput->ensureCursorVisible();
}

void ProcessWidget::addError(const QString &msg)
{
    ui->textOutput->appendPlainText("");
    ui->textOutput->appendHtml(QString("<p style=\"color:%1\">").arg(OUTPUT_CONSOLE_ERROR_COLOR) + msg + "</p>");
    ui->textOutput->appendPlainText("");
    ui->textOutput->ensureCursorVisible();
}

