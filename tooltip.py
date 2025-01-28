# tooltip.py

import tkinter as tk

class ToolTip:
    """
    ToolTip creates a tooltip for a given widget, displaying informative text on hover.
    """

    def __init__(self, widget, text):
        """
        Initialize the ToolTip with the target widget and tooltip text.

        Args:
            widget (tk.Widget): The widget to attach the tooltip to.
            text (str): The tooltip text to display.
        """
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        """
        Display the tooltip when the mouse enters the widget.
        """
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
        
    def hide_tip(self, event=None):
        """
        Hide the tooltip when the mouse leaves the widget.
        """
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None
