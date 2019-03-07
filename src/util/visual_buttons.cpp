//TODO: ELISE HEADER
#if ELISE_QT

#include "general/visual_buttons.h"

cSelectionButton::cSelectionButton(QString text, int id, bool normalize, QWidget *parent) :
     QPushButton(parent),
      _m_line_id(id),
      _m_nrm(normalize)
{
    setText(text);
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
}

void cSelectionButton::onClick()
{
    emit my_click(_m_line_id);
    emit my_click(_m_line_id, _m_nrm);
}

cSpinBox::cSpinBox(int value, QWidget *parent) :
    QSpinBox(parent),
    _m_index(0)
{
    for (int aK=0; aK< 20; ++aK)
    {
        _m_values.push_back((int) pow(2.f, (float) aK));
        if (value == _m_values.back()) _m_index = aK;
    }

    setMinimum(_m_values.at(0));
    setMaximum(_m_values.at(_m_values.size()-1));
    setValue(value);
}

void cSpinBox::stepBy(int steps)
{
    _m_index += steps;
    _m_index = qBound(0, _m_index, _m_values.size() - 1);
    setValue(_m_values.at(_m_index));
}

#endif // ELISE_QT





