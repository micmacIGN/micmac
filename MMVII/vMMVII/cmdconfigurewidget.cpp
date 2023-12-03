#include "cmdconfigurewidget.h"
#include "global.h"

#include <QScrollArea>


CmdConfigureWidget::CmdConfigureWidget(MMVIISpecs &allSpecs, QWidget *parent)
    : QWidget{parent},allSpecs(allSpecs),specs(nullptr),wMandatory(nullptr),wOptional(nullptr),wTuning(nullptr),wGlobal(nullptr)
{
    QVBoxLayout *vLayout = new QVBoxLayout(this);

    teCommand = new QTextEdit(this);
    teCommand->setReadOnly(true);
    QFontMetrics fm(teCommand->font());
    QMargins margins = teCommand->contentsMargins();
    teCommand->setMinimumHeight(fm.height() * 3 + margins.top() + margins.bottom() + 8);
    teCommand->setMaximumHeight(fm.height() * 7 + margins.top() + margins.bottom() + 8);


    tabWidget = new QTabWidget(this);

    vLayout->addWidget(tabWidget);
    vLayout->addWidget(teCommand);
}


void CmdConfigureWidget::setSpecs(CommandSpec* specs, CommandSpec initSpec)
{
    this->specs = specs;

    tabWidget->clear();
    delete wMandatory;
    delete wOptional;
    delete wTuning;
    delete wGlobal;

    wMandatory = createPage(specs->mandatories, "");
    wOptional  = createPage(specs->optionals, "normal");
    wTuning    = createPage(specs->optionals, "tuning");
    wGlobal    = createPage(specs->optionals, "global");

    if (wMandatory)
        tabWidget->addTab(wMandatory,tr("Mandatory"));
    if (wOptional)
        tabWidget->addTab(wOptional,tr("Optional"));
    if (wTuning)
        tabWidget->addTab(wTuning,tr("Tuning"));
    if (wGlobal)
        tabWidget->addTab(wGlobal,tr("Global"));

    if (!initSpec.empty()) {
        specs->initFrom(initSpec);
        InputWidget::initValues();
    }

    resize(width(),minimumHeight());
    checkAllParams();
    updateCommand();
}




InputWidget *CmdConfigureWidget::createInput(QWidget *widget, QGridLayout *layout, ArgSpec& as)
{
    switch (as.cppType) {
    case ArgSpec::T_CHAR:
        return new InputChar(widget, layout, as);
    case ArgSpec::T_ENUM:
    case ArgSpec::T_BOOL:
        return new InputEnum(widget, layout, as);
    case ArgSpec::T_INT:
        return new InputIntN(widget, layout, as, 1);
    case ArgSpec::T_DOUBLE:
        return new InputDoubleN(widget, layout, as, 1);
    case ArgSpec::T_PTXD2_INT:
        return new InputIntN(widget, layout, as, 2);
    case ArgSpec::T_PTXD3_INT:
        return new InputIntN(widget, layout, as, 3);
    case ArgSpec::T_PTXD2_DOUBLE:
        return new InputDoubleN(widget, layout, as, 2);
    case ArgSpec::T_PTXD3_DOUBLE:
        return new InputDoubleN(widget, layout, as, 3);
    case ArgSpec::T_BOX2_INT:
        return new InputIntN(widget, layout, as, 4);
    case ArgSpec::T_VEC_INT:
        if (as.vSizeMax <= 9)
            return new InputIntN(widget, layout, as, 0);
        else
            return new InputString(widget, layout, as);
    case ArgSpec::T_VEC_DOUBLE:
        if (as.vSizeMax <= 9)
            return new InputDoubleN(widget, layout, as, 0);
        else
            return new InputString(widget, layout, as);
    case ArgSpec::T_STRING:
        if (contains(as.semantic,eTA2007::FFI))
            return new InputFFI(widget, layout, as);
        if (contains(as.semantic,{eTA2007::DirProject, eTA2007::FileDirProj,eTA2007::MPatFile, eTA2007::vMMVII_PhpPrjDir,eTA2007::vMMVII_FilesType}))
            return new InputFile(widget, layout, as, allSpecs);
        return new InputString(widget, layout, as);
    case ArgSpec::T_VEC_STRING:
        return new InputStrings(widget,layout,as,0);
    case ArgSpec::T_UNKNOWN:
        return new InputString(widget, layout, as);
    }
    return nullptr;     // should not happen ...
}


QWidget *CmdConfigureWidget::createPage(std::vector<ArgSpec>& argSpecs, const QString& level)
{
    QWidget *widget = new QWidget;
    QGridLayout *layout= new QGridLayout(widget);
    QScrollArea *scroll = new QScrollArea;
    scroll->setWidget(widget);

    widget->setFixedWidth(600);
//    scroll->setMinimumHeight(400);
    layout->setSizeConstraint(QLayout::SetMinAndMaxSize);
    layout->setContentsMargins(5,25,5,5);   // FIXME CM: 25 needed to have tooltip working in the first element ...

    for (auto& as : argSpecs) {
        if (level != as.level)
            continue;
        auto input = createInput(widget,layout, as);
        input->checkValue();
        connect(input, &InputWidget::valueChanged, this, &CmdConfigureWidget::valueUpdated);
    }
    int nb3=0;
    int nb2=0;
    for (int i=0; i<layout->rowCount(); i++) {
        if (layout->itemAtPosition(i,3))
            nb3++;
        if (layout->itemAtPosition(i,2))
            nb2++;
    }
    layout->setColumnStretch(0,0);
    layout->setColumnStretch(1,1);
    layout->setColumnStretch(2,nb2==0 ? 0 : 1);
    layout->setColumnStretch(3,nb3==0 ? 0 : 1);
    layout->setColumnStretch(4,0);
    layout->setRowStretch(layout->rowCount(),1);
    if (layout->count() == 0) {
        delete scroll;
        scroll = nullptr;
    }
    return scroll;
}


void CmdConfigureWidget::resetValues()
{
    InputWidget::resetAll();
}

void CmdConfigureWidget::checkAllParams()
{
    bool ok = true;
    for (const auto& as : specs->mandatories) {
        if (! as.check) {
            ok = false;
            break;
        }
    }
    if (ok) {
        for (const auto& as : specs->optionals) {
            if (! as.check) {
                ok = false;
                break;
            }
        }
    }
    emit canRunSignal(ok);
}



void CmdConfigureWidget::updateCommand()
{
    cmdLine  = specs->name;
    for (const auto& as : specs->mandatories)
        cmdLine += " " + quotedArg(as.value);

    for (const auto& as : specs->optionals) {
        if (as.isEnabled)
            cmdLine += " " + quotedArg(as.name + "=" + as.value);
    }
    teCommand->setText("MMVII " + cmdLine);
}

void CmdConfigureWidget::valueUpdated(const ArgSpec &as)
{
    if (as.mandatory)
        this->checkAllParams();
    this->updateCommand();
}

void CmdConfigureWidget::doRun()
{
    QStringList args(specs->name);
    for (const auto& as : specs->mandatories)
        args.append(as.value);

    for (const auto& as : specs->optionals) {
        if (as.isEnabled)
            args.append(as.name + "=" +  as.value);
    }
    emit runSignal(args);
}

