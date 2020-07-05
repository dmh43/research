(TeX-add-style-hook
 "summary"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "10pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8") ("tocbibind" "nottoc")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "babel"
    "inputenc"
    "tocbibind"
    "amsfonts"
    "amsmath"
    "setspace"
    "hyperref")
   (LaTeX-add-labels
    "eq:general-loss"
    "eq:er"
    "eq:uni-dev")
   (LaTeX-add-environments
    '("Assumption" LaTeX-env-args ["argument"] 0)
    '("assumption" LaTeX-env-args ["argument"] 0)
    '("assumption" LaTeX-env-args ["argument"] 1)
    '("theorem" LaTeX-env-args ["argument"] 0))
   (LaTeX-add-bibliographies))
 :latex)

