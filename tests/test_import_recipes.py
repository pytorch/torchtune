def test_import_receipes():
    import recipes
    print(recipes)
    from recipes.full_finetune_single_device import FullFinetuneRecipeSingleDevice
    print(FullFinetuneRecipeSingleDevice)
