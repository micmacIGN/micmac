#ifndef VISUAL_BUTTONS_H
#define VISUAL_BUTTONS_H

#include "CMake_defines.h"

#if(ELISE_QT_VERSION >= 4)

#ifdef Int
    #undef Int
#endif

#include <QPushButton>

class cSelectionButton: public QPushButton
{
    Q_OBJECT

public:
        cSelectionButton(QString text, int id = -1,  QWidget *parent=0);

protected:
        int  _m_line_id;

public slots:
        void onClick();

signals:
        void my_click(int);

};

#endif //ELISE_QT_VERSION >= 4

#endif /* VISUAL_BUTTONS_H */
