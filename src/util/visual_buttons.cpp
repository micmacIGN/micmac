#include "general/visual_buttons.h"

#if (ELISE_QT_VERSION >= 4)

cSelectionButton::cSelectionButton(QString text, int id, bool normalize, QWidget *parent)
    : QPushButton(parent),
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
#endif // ELISE_QT_VERSION >= 4



