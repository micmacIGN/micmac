#ifndef INPUTWIDGET_H
#define INPUTWIDGET_H

#include <QWidget>
#include <QGridLayout>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include "commandspec.h"
#include "spinboxdefault.h"



class InputWidget : public QWidget
{
    Q_OBJECT
public:
    explicit InputWidget(QWidget *parent, QGridLayout *layout, ArgSpec& as);
    virtual ~InputWidget();

    void reset();
    static void resetAll();
    static void initValues();
    void checkValue();

signals:
    void valueChanged(const ArgSpec &);

protected:
    enum class State {EMPTY, BAD, OK};
    void addWidget(QWidget* w, int span = 1);
    void valueEdited(const QString &val);
    void enableChanged(bool checked);
    void finish();
    virtual void setInitialValue() = 0;
    virtual void doReset() = 0;
    virtual State doCheckValue() = 0;

    QGridLayout *layout;
    ArgSpec& as;
    QCheckBox *enabledWidget;
    QLabel *label;
    QString noValueMarker;
    int curRow;
    int curCol;
    static QVector<InputWidget *> allInputs;
};


class InputEnum : public InputWidget
{
public:
    InputEnum(QWidget *parent, QGridLayout *layout, ArgSpec &as);
protected:
    virtual void doReset() override;
    virtual void setInitialValue() override;
    State doCheckValue() override;
private:
    QComboBox *cb;
};


class InputString: public InputWidget
{
public:
    InputString(QWidget *parent, QGridLayout *layout, ArgSpec &as);
protected:
    virtual void doReset() override;
    virtual void setInitialValue() override;
    State doCheckValue()  override;
    QLineEdit *lineEdit;
};

class InputFFI: public InputString
{
public:
    InputFFI(QWidget *parent, QGridLayout *layout, ArgSpec &as);
};


class InputChar: public InputString
{
public:
    InputChar(QWidget *parent, QGridLayout *layout, ArgSpec &as);
};

class InputFile: public InputWidget
{
public:
    InputFile(QWidget *parent, QGridLayout *layout, ArgSpec &as, const MMVIISpecs &allSpecs);
protected:
    virtual void doReset() override;
    virtual void setInitialValue() override;
    State doCheckValue() override;
private:
    void fileDialog();
    QLineEdit *le;
    QPushButton *pb;

    QString filter,caption;
    QString subdir;
    enum Mode {FILE_MODE, DIR_MODE};
    Mode mode;
};

class InputStrings: public InputWidget
{
public:
    InputStrings(QWidget *parent, QGridLayout *layout, ArgSpec &as, int n);
protected:
    virtual void doReset() override;
    virtual void setInitialValue() override;
    State doCheckValue()  override;
private:
    void valueEdited(const QString& val, int n);
    QVector<QLineEdit*> les;
};


template<typename T, class SPIN>
class InputNumbers : public InputWidget
{
public:
    InputNumbers(QWidget *parent, QGridLayout *layout, ArgSpec &as,int n);
protected:
    virtual void doReset() override;
    virtual void setInitialValue() override;
    State doCheckValue() override;
private:
    void valueEdited(const QString& val, int n);
    QVector<SPIN *> sbs;
};

typedef InputNumbers<double, DoubleSpinBoxDefault> InputDoubleN;
typedef InputNumbers<int, SpinBoxDefault> InputIntN;



/*********************************************************************************/


template<typename T, class SPIN>
InputNumbers<T,SPIN>::InputNumbers(QWidget *parent, QGridLayout *layout, ArgSpec &as, int n)
    : InputWidget(parent,layout,as)
{
    noValueMarker = " ";
    if ( n > 0)
        as.vSizeMin = as.vSizeMax = n;

    for (int i=0; i<as.vSizeMax; i++) {
        SPIN *sb = new SPIN(this);
        sb->setSpecialValueText(noValueMarker);


        T max = std::numeric_limits<T>::max();
        T min = std::numeric_limits<T>::min();

        QDoubleSpinBox *dsb=dynamic_cast<QDoubleSpinBox*>(sb);
        if (dsb) {
            min = -max;
            dsb->setDecimals(3);
        }
        auto range = parseList<StrList>(as.range);

        if (range.size() > 0 && range[0].size())
            min = qvariant_cast<T>(range[0]);
        if (range.size() > 1 && range[1].size())
            max= qvariant_cast<T>(range[1]);
        sb->setRange(min, max);

#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
        connect(sb,qOverload<const QString&>(&SPIN::valueChanged),this,[this,i](const QString& val) {this->valueEdited(val,i);});
#else
        connect(sb,&SPIN::textChanged,this,[this,i](const QString& val) {this->valueEdited(val,i);});
#endif
        addWidget(sb,1);
        sbs.push_back(sb);
        if (as.vSizeMax == 4 && i==1) {
            curCol = 1;
            curRow ++;
        }
    }
    finish();
}


template<typename T, class SPIN>
void InputNumbers<T,SPIN>::doReset()
{
    auto values = parseList<StrList>(as.def);
    for (int i=0; i<as.vSizeMax; i++) {
        if (i<(int)values.size() && !values[i].isEmpty())
            sbs[i]->setValue(qvariant_cast<T>(values[i]));
        else
            sbs[i]->setValue(sbs[i]->minimum());
    }
}

template<typename T, class SPIN>
void InputNumbers<T,SPIN>::setInitialValue()
{
    if (! as.hasInitValue)
        return;
    auto values = parseList<StrList>(as.initValue);
    for (int i=0; i<as.vSizeMax; i++) {
        if (i<(int)values.size() && !values[i].isEmpty()) {
            sbs[i]->setValue(qvariant_cast<T>(values[i]));
        } else {
            sbs[i]->setValue(sbs[i]->minimum());
        }
    }
}

template<typename T, class SPIN>
InputWidget::State InputNumbers<T, SPIN>::doCheckValue()
{
    if (as.vSizeMax == 1)
        return sbs[0]->text().isEmpty() || sbs[0]->text() == noValueMarker ? State::EMPTY : State::OK;

    int nbValued=0;
    bool allEmpty = true;
    bool oneEmpty = false;
    for (int i=0; i<as.vSizeMax; i++) {
        if (sbs[i]->text().isEmpty() || sbs[i]->text() == noValueMarker)
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

template<typename T, class SPIN>
void InputNumbers<T,SPIN>::valueEdited(const QString &, int )
{
    if (doCheckValue() != State::OK) {
        InputWidget::valueEdited("");
        return;
    }
    if (as.vSizeMax == 1) {
        InputWidget::valueEdited(sbs[0]->text());
        return;
    }
    int nbValued=0;
    for (int i=0; i<as.vSizeMax; i++) {
        if (sbs[i]->text().isEmpty() || sbs[i]->text() == noValueMarker)
            break;
        nbValued++;
    }
    if (nbValued < as.vSizeMin) {
        InputWidget::valueEdited("");
        return;
    }
    for (int i=nbValued; i<as.vSizeMax; i++)
        sbs[i]->setValue(sbs[i]->minimum());

    QString s = "[";
    int i;
    for (i=0;i<as.vSizeMax && !sbs[i]->text().isEmpty() && sbs[i]->text() != noValueMarker; i++) {
        if (i != 0)
            s += ",";
        s += sbs[i]->text();
    }
    s += "]";

    InputWidget::valueEdited(s);
}

#endif // INPUTWIDGET_H
