#include "spinboxdefault.h"
#include <QDateTimeEdit>


/*****************************************************************************************/
SpinBoxDefault::SpinBoxDefault(QWidget *parent)
    :QSpinBox(parent)
{
    allowEmpty = true;
    defaultValue = std::numeric_limits<int>::min();
    setMinimumWidth(100);
}


void SpinBoxDefault::setRange(int min, int max)
{
    if (allowEmpty && min != std::numeric_limits<int>::min())
        min--;
    QSpinBox::setRange(min,max);
}

void SpinBoxDefault::setDefaultValue(int def)
{
    defaultValue = def;
}

QValidator::State SpinBoxDefault::validate(QString &input, int &pos) const
{
    if (allowEmpty && input.isEmpty())
        return QValidator::Acceptable;
    return QSpinBox::validate(input,pos);
}

void SpinBoxDefault::stepBy(int steps)
{
    if (allowEmpty && value()==minimum()) {
        if (defaultValue != std::numeric_limits<int>::min()) {
            setValue(defaultValue);
            return;
        }
        if (minimum() == std::numeric_limits<int>::min()) {
            setValue(0);
            return;
        }
    }
    QSpinBox::stepBy(steps);
}

int SpinBoxDefault::valueFromText(const QString &text) const
{
    if (text.isEmpty() && allowEmpty)
        return minimum();
    return QSpinBox::valueFromText(text);
}


/*****************************************************************************************/
DoubleSpinBoxDefault::DoubleSpinBoxDefault(QWidget *parent)
    :QDoubleSpinBox(parent)
{
    allowEmpty = true;
    defaultValue = 0;
    setMinimumWidth(100);
}


void DoubleSpinBoxDefault::setRange(double min, double max)
{
    if (allowEmpty && min != -std::numeric_limits<double>::max())
        min--;
    QDoubleSpinBox::setRange(min,max);
}

void DoubleSpinBoxDefault::setDefaultValue(double def)
{
    defaultValue = def;
}


QValidator::State DoubleSpinBoxDefault::validate(QString &input, int &pos) const
{
    if (allowEmpty && input.isEmpty())
        return QValidator::Acceptable;
    return QDoubleSpinBox::validate(input,pos);
}

void DoubleSpinBoxDefault::stepBy(int steps)
{
    if (allowEmpty && value()==minimum()) {
        if (defaultValue != -std::numeric_limits<double>::max()) {
            setValue(defaultValue);
            return;
        }
        if (minimum() == -std::numeric_limits<double>::max()) {
            setValue(0);
            return;
        }
    }
    QDoubleSpinBox::stepBy(steps);
}

double DoubleSpinBoxDefault::valueFromText(const QString &text) const
{
    if (text.isEmpty() && allowEmpty)
        return minimum();
    return QDoubleSpinBox::valueFromText(text);
}


