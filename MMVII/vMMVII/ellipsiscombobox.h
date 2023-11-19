#ifndef ELLIPSISCOMBOBOX_H
#define ELLIPSISCOMBOBOX_H

#include <QComboBox>

class EllipsisComboBox : public QComboBox
{
public:
    EllipsisComboBox(QWidget *widget) : QComboBox(widget) {}

private:
    void paintEvent(QPaintEvent * /*event*/) override;
};

#endif // ELLIPSISCOMBOBOX_H
