import pytest
import unittest
from portcast_analytics import data_preprocess, get_seasonality_trend_residue, \
    get_model_CV_report, forecast_residue

class ClassTest(unittest.TestCase):

    ## UNIT TEST
    def test_data_preprocess(self):
        df = data_preprocess('sample.csv')
        self.assertGreater(df.shape[0],0)
        self.assertEqual(df.shape[1],2)
        self.assertEqual(df.columns[0],'ds')
        self.assertEqual(df.columns[1],'y')


    ## UNIT TEST
    def test_get_seasonality_trend_residue(self):
        df = data_preprocess('sample.csv')
        inp_shape = df.shape
        sdf, rdf = get_seasonality_trend_residue(df)
        self.assertEqual(sdf.shape,inp_shape)
        self.assertEqual(rdf.shape,inp_shape)

        n=0
        sdf, rdf = get_seasonality_trend_residue(df,n)
        self.assertEqual(rdf.shape[0], rdf.dropna().shape[0])
        self.assertEqual(sdf.shape[0], sdf.dropna().shape[0])

        n=4
        sdf, rdf = get_seasonality_trend_residue(df,n)
        self.assertEqual(rdf.shape[0], n+rdf.dropna().shape[0])
        self.assertEqual(sdf.shape[0], sdf.dropna().shape[0])



    ## FULL INTEGRATION TEST
    def test_forecast_residue(self):
        n=6
        df = data_preprocess('sample.csv')
        sdf, rdf = get_seasonality_trend_residue(df,n)
        self.assertEqual(rdf.shape[0], n+rdf.dropna().shape[0])
        self.assertEqual(sdf.shape[0], sdf.dropna().shape[0])

        model, past_lags, mae_1wk = get_model_CV_report(rdf,regressor_num=1,forecast_k=1)
        model, past_lags, mae_6wk = get_model_CV_report(rdf,regressor_num=1,forecast_k=6)
        self.assertGreater(mae_6wk,mae_1wk)

        residue_df = forecast_residue(rdf, model, past_lags)
        self.assertEqual(residue_df.shape[0], residue_df.dropna().shape[0])
