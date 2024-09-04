Translation from: <https://github.com/CDC-DNPAO/CDCAnthro>

### Description

Generate z-scores, percentiles, and other metrics for weight, height, and BMI based on the 2000 CDC growth charts (Kuczmarski et al., 2002), BMI metrics proposed at a 2018 meeting (Freedman et al., 2019), and extended z-scores and percentiles for children with obesity (Wei et al., 2020). 

The BMI metrics included z-scores and percentiles based on the growth charts and newer metrics that more accurately characterize BMIs above the CDC 97th percentile. Note that the output variables - bmiz and bmip - are based on a combination of the LMS-based z-scores (Cole and Green, 1992; Centers for Disease Control and Prevention (CDC), 2022) for children without obesity and extended bmiz and extended bmip for children with obesity. The LMS-based z-scores/percentiles are named 'original_bmiz' and 'original_bmip'.

The calculations and output variables are similar to those in the SAS program at __<https://www.cdc.gov/nccdphp/dnpao/growthcharts/resources/sas.htm>__. However, using all = 'TRUE' in the function will output additional BMI metrics.



While this is the translation to Python of the original R script, you can find the full details here: <https://github.com/CDC-DNPAO/CDCAnthro/blob/main/README.md> 
