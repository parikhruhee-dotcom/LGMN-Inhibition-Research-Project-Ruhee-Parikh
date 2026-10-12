# The biological premise, and the case against it

This project assumes that inhibiting legumain is a plausible Alzheimer's strategy.
That assumption was carried through the whole pipeline without a citation, which is
the wrong way round: the premise deserves more scrutiny than the docking parameters.

All articles below were retrieved from **PubMed**.

## The premise is well supported

Legumain, also called asparaginyl endopeptidase (AEP) or δ-secretase, cleaves both of
the proteins central to Alzheimer's pathology, which is what makes it a dual-pathway
target rather than a single-pathway one.

- Wu *et al.* (2019), *Progress in Neurobiology* — AEP "cleaves both APP and Tau at APP
  N585 and Tau N368 sites, respectively, which mediate AD pathogenesis by promoting Aβ
  production and Tau hyperphosphorylation". Knocking out AEP reduced AD-like pathology
  and cognitive impairment in the 3xTg mouse model.
  [DOI](https://doi.org/10.1016/j.pneurobio.2019.101730) · PMID 31778772

- Liao *et al.* (2021), *Neuropharmacology* — δ-secretase activation "simultaneously
  cleaves both APP and Tau and promotes Aβ production and neurofibrillary tangle
  pathologies".
  [DOI](https://doi.org/10.1016/j.neuropharm.2021.108737) · PMID 34343610

These confirm the specific cleavage sites this project's README states (tau N368, APP
N585), which had been asserted from memory until now.

## The premise is also contested

A read of the supporting literature alone would be a biased one. At least one study
argues directly against the therapeutic strategy:

- Behrendt *et al.* (2019), *Neurobiology of Disease* — finds that AEP is expressed in
  microglia rather than neurons, that tau cleavage at N368 occurs after uptake into
  microglia, and that the resulting fragment is present "in both control and AD brains
  at similar levels". The authors conclude that AEP-mediated tau cleavage "is a
  physiological event occurring during microglial degradation of the secreted neuronal
  protein" and explicitly **"caution against preventing AEP-mediated tau cleavage as a
  therapeutic approach in AD"**.
  [DOI](https://doi.org/10.1016/j.nbd.2019.104518) · PMID 31229689

This matters more than any result in this repository. If that interpretation is right,
a perfectly selective, brain-penetrant legumain inhibitor would block a normal
clearance process rather than a pathogenic one. No amount of docking, QSAR or ADMET
work downstream can detect that, because every method here assumes inhibition is
desirable and only asks which molecule does it best.

It does not settle the question either — the knockout and inhibitor studies above
report benefit in animal models — but a write-up that cited only the supporting side
would be misleading.

## Brain penetration has been achieved, which tempers this project's conclusion

The central finding here is that published legumain chemistry is too polar and too
covalent to reach the brain. That is true of the ChEMBL set analysed, but it should not
be read as "impossible":

- Liao *et al.* (2021) report an optimised δ-secretase inhibitor (#11a) with
  "demonstrable brain exposure and oral bioavailability", suppressing APP N585 and Tau
  N368 cleavage in vivo and improving cognition in 3xTg mice.
  [DOI](https://doi.org/10.1016/j.neuropharm.2021.108737) · PMID 34343610

So the right conclusion from `docs/SHORTLIST.md` is that brain-penetrant legumain
inhibition is **hard and underrepresented in public bioactivity data**, not that it
cannot be done. A compound achieving it may simply not be in ChEMBL, which is a
limitation of the dataset rather than of the chemistry.

## What this changes about the project

1. The README's cleavage-site claims are now cited rather than asserted.
2. The therapeutic hypothesis is **contested in the literature**, and any write-up
   should say so. This is a genuine scientific risk to the project's premise, and it
   sits upstream of every computational result in this repository.
3. The "almost nothing is brain-penetrant" conclusion is a statement about the public
   dataset, not about what medicinal chemistry can achieve.

## A note on how these were found

Searched via PubMed for the cleavage sites and the therapeutic hypothesis, including a
deliberate search for disconfirming evidence rather than only supporting work. Every
claim above is quoted from the abstract of the cited paper; none is inferred.
