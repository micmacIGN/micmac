#include "inputwidget.h"
#include "global.h"
#include <QFileDialog>
#include <QMessageBox>

QVector<InputWidget *> InputWidget::allInputs;

void InputWidget::resetAll()
{
    for (auto& w : allInputs)
        w->reset();
}

void InputWidget::initValues()
{
    for (auto& w : allInputs)
        w->setInitialValue();
}


/*********************************************************************************/
InputWidget::InputWidget(QWidget *parent, QGridLayout *layout, ArgSpec &as)
    : QWidget(parent), layout(layout),as(as),enabledWidget(nullptr)
{
    curRow = layout->rowCount();
    curCol = 1;
    as.value="";
    as.check=false;
    label = new QLabel();
    label->setTextFormat(Qt::RichText);
    label->setStyleSheet("QLabel { color : red; }");
    QString toolType;
    if (as.mandatory) {
        if (as.comment.isEmpty())
            label->setText(tr("Argument #%1 :").arg(as.number));
        else
            label->setText(as.comment.toHtmlEscaped() + " :");
    } else {
        label->setText(as.name+ " :");
        toolType = "<p style='white-space:pre'>";
        if (as.comment.isEmpty())
            toolType += as.name;
        else
            toolType += as.comment.toHtmlEscaped();
        if (as.def.size())
            toolType += tr("<p>Default: <b>") + as.def.toHtmlEscaped() + "</b>";
    }
    label->setToolTip(toolType);
    label->setStatusTip(as.comment);
    layout->addWidget(label,curRow,0,Qt::AlignRight);

    if (! as.mandatory) {
        enabledWidget = new QCheckBox();
        enabledWidget->setChecked(false);
        enabledWidget->setToolTip("Checked if present on command line");
        layout->addWidget(enabledWidget,curRow,4,Qt::AlignLeft);
        layout->setColumnStretch(4,0);
        connect(enabledWidget,&QCheckBox::toggled,this,&InputWidget::enableChanged);
    }

    allInputs.push_back(this);
}

InputWidget::~InputWidget()
{
    allInputs.removeOne(this);
}


void InputWidget::addWidget(QWidget *w, int span)
{
    w->setParent(this);
    layout->addWidget(w,curRow,curCol,1,span);
    curCol += span;
    if (curCol == 4) {
        curRow++;
        curCol=1;
    }
}

void InputWidget::checkValue()
{
    auto check = doCheckValue();
    as.check = check == State::OK;

    if (enabledWidget && !enabledWidget->isChecked()) {
        label->setStyleSheet(check == State::BAD ? "QLabel { color : Orange; }" : "QLabel { color : Black; }");
        as.check = true;
    } else {
        label->setStyleSheet(check == State::OK ? "QLabel { color : green; }" : "QLabel { color : Brown; }");
    }
    emit valueChanged(as);
}

void InputWidget::valueEdited(const QString &val)
{
    as.value = val;
    if (enabledWidget)
        enabledWidget->setChecked(!val.isEmpty() && val != noValueMarker);
    checkValue();
}

void InputWidget::enableChanged(bool checked)
{
    if (checked && (as.value.isEmpty() || as.value == noValueMarker)) {
        enabledWidget->setChecked(false);
        checkValue();
        return;
    }
    as.isEnabled = checked;
    checkValue();
}

void InputWidget::finish()
{
    if (showDebug)
        label->setToolTip(label->toolTip() + "<pre>" + as.json.toHtmlEscaped() + "</pre>");
    reset();
}

void InputWidget::reset()
{
    doReset();
    if (enabledWidget)
        enabledWidget->setChecked(false);
}

/*********************************************************************************/
InputEnum::InputEnum(QWidget *parent, QGridLayout *layout, ArgSpec &as) : InputWidget(parent,layout,as)
{
    noValueMarker = " ";
    cb=new QComboBox();
    addWidget(cb,2);
    cb->show();
    if (! as.mandatory)
        cb->addItem(noValueMarker);
    if (as.cppType == ArgSpec::T_BOOL) {
        cb->addItems({"false","true"});
    } else {
        for (const auto& allowed: as.allowed)
            cb->addItem(allowed);
    }

    connect(cb,&QComboBox::currentTextChanged,this,[this](const QString& text) {this->valueEdited(text);});
    finish();
}

void InputEnum::doReset()
{
    if (as.def.isEmpty())
        cb->setCurrentIndex(0);
    else
        cb->setCurrentText(as.def);
    as.value = cb->currentText();
}

void InputEnum::setInitialValue()
{
    if (as.hasInitValue) {
        cb->setCurrentText(as.initValue);
        as.value = cb->currentText();
    }
}

InputWidget::State InputEnum::doCheckValue()
{
    return as.value.isEmpty() ? State::EMPTY : State::OK;
}

/*********************************************************************************/
InputString::InputString(QWidget *parent, QGridLayout *layout, ArgSpec &as) : InputWidget(parent,layout,as)
{
    lineEdit = new QLineEdit;
    addWidget(lineEdit,2);
    connect(lineEdit,&QLineEdit::textChanged,this,[this](const QString& val) {this->valueEdited(val);});
    if (showDebug && as.cppType == ArgSpec::T_UNKNOWN) {
        QLabel *l = new QLabel();
        l->setPixmap(style()->standardIcon(QStyle::SP_MessageBoxWarning).pixmap(16,16));
        addWidget(l,3);
    }
    finish();
}

void InputString::doReset()
{
    lineEdit->setText(as.def);
}

void InputString::setInitialValue()
{
    if (as.hasInitValue)
        lineEdit->setText(as.initValue);
}

InputWidget::State InputString::doCheckValue()
{
    if (as.value.isEmpty())
        return State::EMPTY;
    else if (lineEdit->hasAcceptableInput())
        return State::OK;
    else
        return State::BAD;
}

/*********************************************************************************/
InputFFI::InputFFI(QWidget *parent, QGridLayout *layout, ArgSpec &as) : InputString(parent,layout,as)
{
    QRegularExpression re("[][].+,.+[][]");
    lineEdit->setValidator(new QRegularExpressionValidator(re,this));
    label->setToolTip(label->toolTip() + tr("\nFormat: <b>'['or ']' FileMin ',' FileMax ']' or '['</b>"));
}


/*********************************************************************************/
InputChar::InputChar(QWidget *parent, QGridLayout *layout, ArgSpec &as) : InputString(parent,layout,as)
{
    lineEdit->setMaxLength(1);
    lineEdit->setMinimumWidth(50);
    lineEdit->setMaximumWidth(50);
}


/*********************************************************************************/

static QString extList2Ext(const StrList& extList)
{
    QString extensions;
    for (const auto& e : extList)
        extensions += ( (e.length()>=1 && e[0] == '.') ? "*" : "" ) + e.toLower() + " ";
    for (const auto& e : extList)
        extensions += ( (e.length()>=1 && e[0] == '.') ? "*" : "" ) + e.toUpper() + " ";
    return extensions;
}

InputFile::InputFile(QWidget *parent, QGridLayout *layout, ArgSpec &as, const MMVIISpecs &allSpecs)
    : InputWidget(parent,layout,as)
{
    le = new QLineEdit();
    addWidget(le,2);
    pb = new QPushButton(tr(""));
    addWidget(pb,1);
    connect(le,&QLineEdit::textChanged,this,[this](const QString& val) {this->valueEdited(val);});
    connect(pb,&QPushButton::clicked,this,&InputFile::fileDialog);
    
    if (contains(as.semantic, eTA2007::vMMVII_PhpPrjDir)) {
        mode = DIR_MODE;
        caption = tr("Select a directory");
        pb->setText(tr("Select Dir"));
        subdir = allSpecs.phpDir + as.phpPrjDir;
        filter  = "";
        if (! QDir(subdir).exists()) {
            if (contains(as.semantic,{eTA2007::OptionalExist,eTA2007::Output})) {
                pb->hide();
            } else {
                pb->setText(tr("No Dir !"));
                pb->setToolTip(tr("Sub directory '%1' doesn't exist").arg(subdir));
                pb->setEnabled(false);
            }
        }
        finish();
        return;
    }

    if (contains(as.semantic, eTA2007::MPatFile)) {
        mode    = FILE_MODE;
        caption = tr("Select a file");
        pb->setText(tr("Set of inputs"));
        filter  = tr("File names set (*.xml;*.json);;") + tr("All")+ "(*)";
        finish();
        return;
    }

    if (contains(as.semantic, eTA2007::vMMVII_FilesType)) {
        mode    = FILE_MODE;
        caption = tr("Select a file");
        pb->setText(tr("Select File"));
        filter  = "(" + extList2Ext(allSpecs.extensions.value(as.fileType)) + ");;" + tr("All")+ "(*)";
        finish();
        return;
    }
    
    if (contains(as.semantic, eTA2007::DirProject)) {
        mode    = DIR_MODE;
        caption = tr("Select a project directory");
        pb->setText(tr("Select Dir"));
        filter  = "";
        finish();
        return;
    }
    
    
    mode    = FILE_MODE;
    caption = tr("Select a file");
    pb->setText(tr("Select File"));
    filter  = "All(*)";
    finish();
    return;
}

void InputFile::doReset()
{
    le->setText(as.def);
}

void InputFile::setInitialValue()
{
    if (as.hasInitValue)
        le->setText(as.initValue);
}

InputWidget::State InputFile::doCheckValue()
{
    if (as.value.isEmpty())
        return State::EMPTY;
    if (contains(as.semantic,eTA2007::MPatFile)) {
        QRegularExpression re(as.value);
        if (! re.isValid())
            return State::BAD;
    }
    return State::OK;
}

void InputFile::fileDialog()
{
    QString fileName,dirName;
    QString openDir;
    QStringList fileNames;

    openDir = subdir;
    if (subdir.isEmpty())
        openDir = QDir::currentPath();

    switch (mode) {
    case FILE_MODE:
        if (contains(as.semantic,{eTA2007::Output,eTA2007::OptionalExist}))
            fileName = QFileDialog::getSaveFileName(this,caption,openDir,filter,nullptr,contains(as.semantic,eTA2007::OptionalExist) ? QFileDialog::DontConfirmOverwrite | QFileDialog::DontUseNativeDialog : QFileDialog::DontUseNativeDialog);
        else
            fileName = QFileDialog::getOpenFileName(this,caption,openDir,filter,nullptr, QFileDialog::DontUseNativeDialog);
        if (fileName.isEmpty())
            return;
        fileName = QDir().relativeFilePath(fileName);
        le->setText(fileName);
        break;
    case DIR_MODE:
        dirName = QFileDialog::getExistingDirectory(this,caption,openDir,QFileDialog::DontUseNativeDialog | QFileDialog::ShowDirsOnly);
        if (dirName.isEmpty())
            return;
        dirName = QDir(openDir).relativeFilePath(dirName);
        le->setText(dirName);
        break;
    default:
        QMessageBox::warning(this,"Unimplemented","Unimplemented (yet!) for this semantic");
        break;
    }
}




InputStrings::InputStrings(QWidget *parent, QGridLayout *layout, ArgSpec &as, int n)
    : InputWidget(parent,layout,as)
{
    if ( n > 0)
        as.vSizeMin = as.vSizeMax = n;

    for (int i=0; i<as.vSizeMax; i++) {
        QLineEdit *le = new QLineEdit(this);
        connect(le,&QLineEdit::textChanged,this,[this,i](const QString& val) {this->valueEdited(val,i);});
        addWidget(le,1);
        les.push_back(le);
        if (as.vSizeMax == 4 && i==1) {
            curCol = 1;
            curRow ++;
        }
    }
    finish();
}



void InputStrings::doReset()
{
    auto values = parseList<StrList>(as.def);
    for (int i=0; i<as.vSizeMax; i++) {
        if (i<(int)values.size() && !values[i].isEmpty())
            les[i]->setText(values[i]);
        else
            les[i]->setText("");
    }
}

void InputStrings::setInitialValue()
{
    if (! as.hasInitValue)
        return;
    auto values = parseList<StrList>(as.initValue);
    for (int i=0; i<as.vSizeMax; i++) {
        if (i<(int)values.size() && !values[i].isEmpty()) {
            les[i]->setText(values[i]);
        } else {
            les[i]->setText("");
        }
    }
}

InputWidget::State InputStrings::doCheckValue()
{
    if (as.vSizeMax == 1)
        return les[0]->text().isEmpty() ? State::EMPTY : State::OK;

    int nbValued=0;
    bool allEmpty = true;
    bool oneEmpty = false;
    for (int i=0; i<as.vSizeMax; i++) {
        if (les[i]->text().isEmpty())
            oneEmpty = true;
        else
            allEmpty = false;
        if (! oneEmpty)
            nbValued++;
    }
    if (nbValued < as.vSizeMin)
        return allEmpty ? State::EMPTY : State::BAD;
    return State::OK;
}

void InputStrings::valueEdited(const QString &, int )
{
    if (doCheckValue() != State::OK) {
        InputWidget::valueEdited("");
        return;
    }
    if (as.vSizeMax == 1) {
        InputWidget::valueEdited(les[0]->text());
        return;
    }
    int nbValued=0;
    for (int i=0; i<as.vSizeMax; i++) {
        if (les[i]->text().isEmpty())
            break;
        nbValued++;
    }
    if (nbValued < as.vSizeMin) {
        InputWidget::valueEdited("");
        return;
    }
    for (int i=nbValued; i<as.vSizeMax; i++)
        les[i]->setText("");

    QString s = "[";
    int i;
    for (i=0;i<as.vSizeMax && !les[i]->text().isEmpty(); i++) {
        if (i != 0)
            s += ",";
        s += les[i]->text();
    }
    s += "]";

    InputWidget::valueEdited(s);
}

