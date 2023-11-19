#include "ellipsiscombobox.h"

#include <QStylePainter>

void EllipsisComboBox::paintEvent(QPaintEvent * /*event*/)
{
    QStyleOptionComboBox opt;
    initStyleOption(&opt);

    QStylePainter p(this);
    p.drawComplexControl(QStyle::CC_ComboBox, opt);

    QRect textRect = style()->subControlRect(QStyle::CC_ComboBox, &opt, QStyle::SC_ComboBoxEditField, this);
    opt.currentText = p.fontMetrics().elidedText(opt.currentText, Qt::ElideMiddle, textRect.width());
    p.drawControl(QStyle::CE_ComboBoxLabel, opt);
}
