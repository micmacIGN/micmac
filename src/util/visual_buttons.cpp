#include "general/visual_buttons.h"

#if (ELISE_QT_VERSION >= 4)

cSelectionButton::cSelectionButton(QString text, int id, QWidget *parent)
    : QPushButton(parent),
      _m_line_id(id)
{
    setText(text);
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
}

void cSelectionButton::onClick()
{
    emit my_click(_m_line_id);
}
#endif // ELISE_QT_VERSION >= 4



