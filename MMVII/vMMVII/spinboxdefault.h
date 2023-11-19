#ifndef SPINBOXDEFAULT_H
#define SPINBOXDEFAULT_H

#include <qspinbox.h>

class SpinBoxDefault : public QSpinBox
{
    Q_OBJECT
public:
    SpinBoxDefault(QWidget *parent = nullptr);
    void setRange(int min,int max);
    void setDefaultValue(int def);
protected:
    QValidator::State validate(QString &input, int &pos) const override;
    void stepBy(int steps) override;

    int valueFromText(const QString &text) const override;
private:
    bool allowEmpty;
    int defaultValue;
};


class DoubleSpinBoxDefault : public QDoubleSpinBox
{
    Q_OBJECT
public:
    DoubleSpinBoxDefault(QWidget *parent = nullptr);
    void setRange(double min,double max);
    void setDefaultValue(double def);
protected:
    QValidator::State validate(QString &input, int &pos) const override;
    void stepBy(int steps) override;

    double valueFromText(const QString &text) const override;
private:
    bool allowEmpty;
    double defaultValue;
};

#endif // SPINBOXDEFAULT_H
